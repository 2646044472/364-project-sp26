from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.detector import (
    BirdDataset,
    build_prediction_label_map,
    confidence_from_probs,
    data_loader_kwargs,
    load_records,
    pad_boxes,
    predict_with_tta,
    set_seed,
    unletterbox_box,
    write_prediction_csv,
)
from src.train import collate_fn
from src.train_crop_feature_head import CropFeatureHead, attention_feature_vector, load_base_model


def load_head(head_path: Path, device: torch.device) -> tuple[CropFeatureHead, dict]:
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    head = CropFeatureHead(
        feature_dim=int(checkpoint["feature_dim"]),
        num_classes=len(checkpoint["idx_to_label"]),
        dropout=0.0,
        hidden_dim=int(checkpoint.get("args", {}).get("hidden_dim", 0)),
    ).to(device)
    state_dict = checkpoint["model"]
    try:
        head.load_state_dict(state_dict)
    except RuntimeError:
        legacy_state_dict = {
            "classifier.0.weight": state_dict["norm.weight"],
            "classifier.0.bias": state_dict["norm.bias"],
            "classifier.2.weight": state_dict["classifier.weight"],
            "classifier.2.bias": state_dict["classifier.bias"],
        }
        head.load_state_dict(legacy_state_dict)
    head.eval()
    return head, checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final T1 predictions from the detector + crop-feature head.")
    parser.add_argument("--data-dir", required=True, help="Birdwatching dataset root.")
    parser.add_argument("--base-checkpoint", required=True, help="Detector checkpoint, e.g. epoch_002.pt.")
    parser.add_argument("--head-checkpoint", required=True, help="Crop-feature head checkpoint, usually best.pt.")
    parser.add_argument("--output-csv", required=True, help="Submission CSV path.")
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", "--random-seed", dest="seed", type=int, default=364)
    parser.add_argument("--score-mode", default=None, choices=["class", "margin", "max_margin", "entropy"])
    parser.add_argument("--blend-weight", type=float, default=None, help="Override saved best_blend_weight.")
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--max-gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records for smoke testing.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        if args.max_gpu_memory_fraction <= 0 or args.max_gpu_memory_fraction > 1:
            raise ValueError("--max-gpu-memory-fraction must be in the range (0, 1].")
        if args.max_gpu_memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction, cuda_index)

    channels_last = device.type == "cuda" and not args.no_channels_last
    base_model, base_checkpoint, image_size = load_base_model(Path(args.base_checkpoint), device)
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    base_model.eval()

    head, head_checkpoint = load_head(Path(args.head_checkpoint), device)
    head_args = head_checkpoint.get("args", {})
    crop_size = int(head_args.get("crop_size", 448))
    crop_padding = float(head_args.get("crop_padding", 0.10))
    blend_weight = float(
        args.blend_weight
        if args.blend_weight is not None
        else head_checkpoint.get("metrics", {}).get("best_blend_weight", 0.35)
    )
    score_mode = args.score_mode or head_args.get("score_mode", "margin")

    data_dir = Path(args.data_dir)
    records = load_records(data_dir, args.split)
    if args.max_records is not None:
        records = records[: max(args.max_records, 0)]
    dataset = BirdDataset(records, None, image_size, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **data_loader_kwargs(args.num_workers, args.prefetch_factor, device, pin_memory=device.type == "cuda"),
    )
    prediction_labels = build_prediction_label_map(data_dir, base_checkpoint["idx_to_label"])
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    image_ids: list[str] = []
    labels: list[str] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    offset = 0
    for batch in tqdm(loader, desc=f"predict {args.split} final"):
        images = batch["image"].to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            full_probs, pred_boxes = predict_with_tta(base_model, images, hflip=args.hflip_tta)
            crops = predict_crops(images, pred_boxes, crop_padding, crop_size)
            crop_features = attention_feature_vector(base_model, crops)
            crop_probs = head(crop_features.float()).softmax(dim=1)
            probs = blend_weight * full_probs.float() + (1.0 - blend_weight) * crop_probs.float()
        confidence_mode = "max" if score_mode == "class" else score_mode
        conf, pred_labels = confidence_from_probs(probs.float().cpu(), mode=confidence_mode)
        pred_boxes_np = pred_boxes.float().cpu().numpy()
        batch_ids = batch["image_id"]
        for batch_index, image_id in enumerate(batch_ids):
            record = records[offset + batch_index]
            image_ids.append(image_id)
            labels.append(prediction_labels[int(pred_labels[batch_index])])
            boxes.append(unletterbox_box(pred_boxes_np[batch_index], record.image_path, image_size))
            scores.append(float(conf[batch_index]))
        offset += len(batch_ids)

    write_prediction_csv(
        Path(args.output_csv),
        image_ids=image_ids,
        labels=labels,
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
    )


def predict_crops(images: torch.Tensor, pred_boxes: torch.Tensor, crop_padding: float, crop_size: int) -> torch.Tensor:
    from src.detector import roi_align_single

    crop_boxes = pad_boxes(pred_boxes.detach(), padding=crop_padding)
    return roi_align_single(images, crop_boxes, crop_size)


if __name__ == "__main__":
    main()
