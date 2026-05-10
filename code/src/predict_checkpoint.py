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
    build_detector,
    confidence_from_probs,
    data_loader_kwargs,
    load_records,
    predict_with_tta,
    set_seed,
    unletterbox_box,
    write_prediction_csv,
)
from src.train import collate_fn


def load_model(checkpoint_path: Path, device: torch.device, *, channels_last: bool) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    idx_to_label = checkpoint["idx_to_label"]
    model = build_detector(
        num_classes=len(idx_to_label),
        pretrained=False,
        backbone_name=args.get("backbone", "efficientnet_b0"),
        detector_type=args.get("detector_type", "attention"),
        hierarchy_num_classes=len(checkpoint.get("idx_to_species", [])) or None,
        crop_size=int(args.get("crop_size", 260)),
        crop_padding=float(args.get("crop_padding", 0.18)),
        crop_full_logit_weight=float(args.get("crop_full_logit_weight", 0.40)),
        hf_model_id=args.get("hf_model_id", "dennisjooo/Birds-Classifier-EfficientNetB2"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle-style CSV predictions from a detector checkpoint.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", "--random-seed", dest="seed", type=int, default=364)
    parser.add_argument("--score-mode", default="max_margin", choices=["class", "margin", "max_margin", "entropy"])
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--crop-classify", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--crop-full-weight", type=float, default=0.20)
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
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device, channels_last=channels_last)
    image_size = int(checkpoint.get("args", {}).get("image_size", 320))
    idx_to_label = checkpoint["idx_to_label"]
    prediction_labels = build_prediction_label_map(Path(args.data_dir), idx_to_label)

    records = load_records(Path(args.data_dir), args.split)
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

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    image_ids: list[str] = []
    labels: list[str] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    offset = 0

    for batch in tqdm(loader, desc=f"predict {args.split}"):
        images = batch["image"].to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            probs, pred_boxes = predict_with_tta(
                model,
                images,
                hflip=args.hflip_tta,
                crop_classify=args.crop_classify,
                crop_padding=args.crop_padding,
                crop_size=args.crop_size,
                crop_full_weight=args.crop_full_weight,
            )
        confidence_mode = "max" if args.score_mode == "class" else args.score_mode
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


if __name__ == "__main__":
    main()
