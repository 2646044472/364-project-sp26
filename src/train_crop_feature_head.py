from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from src.detector import (
    BirdDataset,
    build_prediction_label_map,
    build_detector,
    confidence_from_probs,
    count_parameters,
    data_loader_kwargs,
    load_records,
    pad_boxes,
    predict_with_tta,
    roi_align_single,
    set_seed,
    unletterbox_box,
)
from src.train import collate_fn
from src.train_teacher_crop_classifier import build_teacher


class CropFeatureHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.15, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def load_calculate_ap(path: Path):
    spec = importlib.util.spec_from_file_location("calculate_AP", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_AP


def load_base_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    idx_to_label = checkpoint["idx_to_label"]
    model = build_detector(
        len(idx_to_label),
        pretrained=False,
        backbone_name=args.get("backbone", "efficientnet_b0"),
        detector_type=args.get("detector_type", "attention"),
        hierarchy_num_classes=len(checkpoint.get("idx_to_species", [])) or None,
        crop_size=int(args.get("crop_size", 260)),
        hf_model_id=args.get("hf_model_id", "dennisjooo/Birds-Classifier-EfficientNetB2"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    image_size = int(args.get("image_size", 320))
    return model, checkpoint, image_size


def attention_feature_vector(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    feats = model.features(images)
    batch, _channels, height, width = feats.shape
    weights = torch.softmax(model.attention(feats).flatten(2), dim=-1)
    pooled = (feats.flatten(2) * weights).sum(dim=-1)

    y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
    x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
    center = (coords * weights).sum(dim=-1)
    spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
    return torch.cat([pooled, center, spread], dim=1)


def cache_path_for(
    output_dir: Path,
    split: str,
    checkpoint_path: Path,
    hflip: bool,
    image_size: int,
    crop_size: int,
    crop_padding: float,
    teacher_key: str = "noteacher",
) -> Path:
    stem = checkpoint_path.parent.name + "_" + checkpoint_path.stem
    tta = "hflip" if hflip else "plain"
    padding_key = f"pad{int(round(crop_padding * 1000)):03d}"
    return output_dir / f"crop_feature_cache_{split}_{stem}_{image_size}_crop{crop_size}_{padding_key}_{tta}_{teacher_key}.pt"


@torch.no_grad()
def collect_crop_features(
    *,
    data_dir: Path,
    split: str,
    checkpoint_path: Path,
    model: nn.Module,
    checkpoint: dict,
    image_size: int,
    crop_size: int,
    crop_padding: float,
    teacher: nn.Module | None = None,
    teacher_crop_size: int = 448,
    teacher_hflip: bool = False,
    teacher_blend_detector_weight: float = 0.15,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    hflip: bool,
    num_workers: int,
    prefetch_factor: int,
    channels_last: bool,
    max_records: int | None = None,
) -> dict[str, object]:
    teacher_key = "noteacher" if teacher is None else f"teacher{teacher_crop_size}_w{int(round(teacher_blend_detector_weight * 100)):02d}"
    cache_path = cache_path_for(output_dir, split, checkpoint_path, hflip, image_size, crop_size, crop_padding, teacher_key)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    records = load_records(data_dir, split)
    if max_records is not None:
        records = records[:max_records]
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = checkpoint["idx_to_label"]
    dataset = BirdDataset(records, label_to_idx, image_size, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **data_loader_kwargs(num_workers, prefetch_factor, device, pin_memory=False),
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    feature_list: list[torch.Tensor] = []
    full_feature_list: list[torch.Tensor] = []
    full_prob_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []
    pred_box_list: list[torch.Tensor] = []
    true_box_list: list[torch.Tensor] = []
    teacher_prob_list: list[torch.Tensor] = []
    image_ids: list[str] = []
    image_paths: list[str] = []
    record_offset = 0

    model.eval()
    if teacher is not None:
        teacher.eval()
    for batch in tqdm(loader, desc=f"cache {split} crop features"):
        images = batch["image"].to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            full_probs, pred_boxes = predict_with_tta(model, images, hflip=hflip)
            full_features = attention_feature_vector(model, images)
            crop_boxes = pad_boxes(pred_boxes.detach(), padding=crop_padding)
            crops = roi_align_single(images, crop_boxes, crop_size)
            crop_features = attention_feature_vector(model, crops)
            if teacher is not None:
                teacher_crops = roi_align_single(images, crop_boxes, teacher_crop_size)
                teacher_logits = teacher(teacher_crops).float()
                teacher_probs = teacher_logits.softmax(dim=1)
                if teacher_hflip:
                    teacher_flip_logits = teacher(torch.flip(teacher_crops, dims=[3])).float()
                    teacher_probs = (teacher_probs + teacher_flip_logits.softmax(dim=1)) * 0.5
                blend_weight = max(0.0, min(1.0, teacher_blend_detector_weight))
                teacher_probs = blend_weight * full_probs.float() + (1.0 - blend_weight) * teacher_probs
                teacher_prob_list.append(teacher_probs.float().cpu())
        feature_list.append(crop_features.float().cpu())
        full_feature_list.append(full_features.float().cpu())
        full_prob_list.append(full_probs.float().cpu())
        label_list.append(batch["label"].cpu())
        pred_box_list.append(pred_boxes.float().cpu())
        true_box_list.append(batch["box_xyxy"].cpu())
        batch_count = len(batch["image_id"])
        image_ids.extend(batch["image_id"])
        image_paths.extend(str(record.image_path) for record in records[record_offset : record_offset + batch_count])
        record_offset += batch_count

    payload = {
        "features": torch.cat(feature_list, dim=0),
        "full_features": torch.cat(full_feature_list, dim=0),
        "full_probs": torch.cat(full_prob_list, dim=0),
        "labels": torch.cat(label_list, dim=0),
        "pred_boxes": torch.cat(pred_box_list, dim=0),
        "true_boxes": torch.cat(true_box_list, dim=0),
        "teacher_probs": torch.cat(teacher_prob_list, dim=0) if teacher_prob_list else None,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "idx_to_label": idx_to_label,
        "image_size": image_size,
        "crop_size": crop_size,
        "crop_padding": crop_padding,
        "teacher_key": teacher_key,
        "split": split,
    }
    torch.save(payload, cache_path)
    return payload


def merge_feature_caches(caches: list[dict[str, object]], split_name: str = "trainval") -> dict[str, object]:
    if not caches:
        raise ValueError("merge_feature_caches requires at least one cache")
    merged = dict(caches[0])
    tensor_keys = ["features", "full_probs", "labels", "pred_boxes", "true_boxes"]
    if all("full_features" in cache for cache in caches):
        tensor_keys.append("full_features")
    for key in tensor_keys:
        merged[key] = torch.cat([cache[key] for cache in caches], dim=0)
    if all(cache.get("teacher_probs") is not None for cache in caches):
        merged["teacher_probs"] = torch.cat([cache["teacher_probs"] for cache in caches], dim=0)
    else:
        merged["teacher_probs"] = None
    for key in ["image_ids", "image_paths"]:
        values: list[str] = []
        for cache in caches:
            values.extend(cache[key])
        merged[key] = values
    merged["split"] = split_name
    return merged


def candidate_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_probs: torch.Tensor,
    *,
    topk: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Focus classification on the confusing top-K bird labels for each image."""
    num_classes = logits.shape[1]
    k = max(1, min(int(topk), num_classes))
    candidate_idx = candidate_probs.topk(k, dim=1).indices
    label_idx = labels.unsqueeze(1)
    candidate_idx = torch.cat([candidate_idx, label_idx], dim=1)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, candidate_idx, True)
    masked_logits = logits.masked_fill(~mask, -1e4)
    rank_ce = nn.functional.cross_entropy(masked_logits, labels)

    if margin <= 0:
        return rank_ce, logits.new_zeros(())

    negative_mask = mask.clone()
    negative_mask.scatter_(1, label_idx, False)
    valid = negative_mask.any(dim=1)
    if not bool(valid.any()):
        return rank_ce, logits.new_zeros(())
    true_logits = logits.gather(1, label_idx).squeeze(1)
    hardest_negative = logits.masked_fill(~negative_mask, -1e4).max(dim=1).values
    margin_loss = nn.functional.relu(float(margin) + hardest_negative[valid] - true_logits[valid]).mean()
    return rank_ce, margin_loss


def ap_from_cache(
    *,
    cache: dict[str, object],
    data_dir: Path,
    probs: torch.Tensor,
    score_mode: str,
) -> float:
    scores, pred_labels = confidence_from_probs(probs.float().cpu(), mode="max" if score_mode == "class" else score_mode)
    labels = cache["labels"].cpu()
    image_size = int(cache["image_size"])
    image_paths = cache["image_paths"]
    pred_boxes = cache["pred_boxes"].numpy()
    true_boxes = cache["true_boxes"].numpy()
    det_boxes = []
    det_labels = []
    det_scores = []
    gt_boxes = []
    gt_labels = []
    for index, image_path_raw in enumerate(image_paths):
        image_path = Path(image_path_raw)
        original_box = unletterbox_box(pred_boxes[index], image_path, image_size).tolist()
        original_true_box = unletterbox_box(true_boxes[index], image_path, image_size).tolist()
        det_boxes.append(torch.tensor([original_box], dtype=torch.float32))
        det_labels.append(torch.tensor([int(pred_labels[index]) + 1], dtype=torch.long))
        det_scores.append(torch.tensor([float(scores[index])], dtype=torch.float32))
        gt_boxes.append(torch.tensor([original_true_box], dtype=torch.float32))
        gt_labels.append(torch.tensor([int(labels[index]) + 1], dtype=torch.long))
    calc = load_calculate_ap(data_dir / "calculate_AP.py")
    _, m_ap = calc(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, probs.shape[1] + 1)
    return float(m_ap)


@torch.no_grad()
def evaluate_head(
    *,
    head: CropFeatureHead,
    cache: dict[str, object],
    data_dir: Path,
    device: torch.device,
    blend_weights: list[float],
    score_mode: str,
) -> dict[str, float]:
    head.eval()
    features = cache["features"].to(device)
    labels = cache["labels"].cpu()
    full_probs = cache["full_probs"].float()
    crop_probs = head(features).softmax(dim=1).float().cpu()
    base_pred = full_probs.argmax(dim=1)
    crop_pred = crop_probs.argmax(dim=1)
    result = {
        "base_accuracy": float((base_pred == labels).float().mean().item()),
        "crop_accuracy": float((crop_pred == labels).float().mean().item()),
        "base_mAP50": ap_from_cache(cache=cache, data_dir=data_dir, probs=full_probs, score_mode=score_mode),
    }
    teacher_probs = cache.get("teacher_probs")
    if teacher_probs is not None:
        teacher_probs = teacher_probs.float()
        teacher_pred = teacher_probs.argmax(dim=1)
        result["teacher_accuracy"] = float((teacher_pred == labels).float().mean().item())
        result["teacher_mAP50"] = ap_from_cache(cache=cache, data_dir=data_dir, probs=teacher_probs, score_mode=score_mode)
    best_blend = 0.0
    best_ap = -1.0
    best_accuracy = 0.0
    for blend_weight in blend_weights:
        probs = blend_weight * full_probs + (1.0 - blend_weight) * crop_probs
        pred = probs.argmax(dim=1)
        accuracy = float((pred == labels).float().mean().item())
        m_ap = ap_from_cache(cache=cache, data_dir=data_dir, probs=probs, score_mode=score_mode)
        result[f"blend_{blend_weight:.2f}_accuracy"] = accuracy
        result[f"blend_{blend_weight:.2f}_mAP50"] = m_ap
        if m_ap > best_ap:
            best_ap = m_ap
            best_blend = blend_weight
            best_accuracy = accuracy
    result["best_blend_weight"] = float(best_blend)
    result["best_blend_accuracy"] = float(best_accuracy)
    result["best_mAP50"] = float(best_ap)
    return result


def write_prediction_csv(
    *,
    data_dir: Path,
    checkpoint_path: Path,
    head_path: Path,
    output_csv: Path,
    batch_size: int,
    device: torch.device,
    hflip: bool,
    score_mode: str,
    num_workers: int,
    prefetch_factor: int,
    channels_last: bool,
) -> None:
    base_model, checkpoint, image_size = load_base_model(checkpoint_path, device)
    if channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    base_model.eval()
    head_checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    head_args = head_checkpoint["args"]
    head = CropFeatureHead(
        feature_dim=int(head_checkpoint["feature_dim"]),
        num_classes=len(checkpoint["idx_to_label"]),
        dropout=0.0,
        hidden_dim=int(head_checkpoint["args"].get("hidden_dim", 0)),
    ).to(device)
    head.load_state_dict(head_checkpoint["model"])
    head.eval()
    crop_size = int(head_args["crop_size"])
    crop_padding = float(head_args["crop_padding"])
    blend_weight = float(head_checkpoint["metrics"]["best_blend_weight"])

    records = load_records(data_dir, "test")
    dataset = BirdDataset(records, None, image_size, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **data_loader_kwargs(num_workers, prefetch_factor, device, pin_memory=False),
    )
    rows: list[dict[str, object]] = []
    idx_to_label = checkpoint["idx_to_label"]
    prediction_labels = build_prediction_label_map(data_dir, idx_to_label)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict crop-feature test"):
            images = batch["image"].to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last if channels_last else torch.contiguous_format,
            )
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                full_probs, pred_boxes = predict_with_tta(base_model, images, hflip=hflip)
                crops = roi_align_single(images, pad_boxes(pred_boxes.detach(), padding=crop_padding), crop_size)
                features = attention_feature_vector(base_model, crops)
                crop_probs = head(features.float()).softmax(dim=1)
                probs = blend_weight * full_probs.float() + (1.0 - blend_weight) * crop_probs.float()
            scores, pred_labels = confidence_from_probs(probs.float().cpu(), mode="max" if score_mode == "class" else score_mode)
            boxes = pred_boxes.float().cpu().numpy()
            for offset, image_id in enumerate(batch["image_id"]):
                original_box = unletterbox_box(boxes[offset], records[len(rows)].image_path, image_size)
                rows.append(
                    {
                        "Id": image_id,
                        "Predictions": {
                            "label": prediction_labels[int(pred_labels[offset])],
                            "score": float(scores[offset]),
                            "xmin": float(original_box[0]),
                            "ymin": float(original_box[1]),
                            "xmax": float(original_box[2]),
                            "ymax": float(original_box[3]),
                        },
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Id", "Predictions"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a legal crop-feature head on frozen B2 detector features.")
    parser.add_argument("--data-dir", default="birdwatching")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--predict-batch-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--crop-padding", type=float, default=0.18)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-model-name", default=None)
    parser.add_argument("--teacher-crop-size", type=int, default=None)
    parser.add_argument("--teacher-hflip-tta", action="store_true")
    parser.add_argument("--teacher-blend-detector-weight", type=float, default=0.15)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for ranking the true label above top-K confusing labels.",
    )
    parser.add_argument(
        "--rank-topk",
        type=int,
        default=12,
        help="Top-K labels from teacher/full probabilities used by the rank loss.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.15,
        help="Logit margin for true label versus hardest top-K negative.",
    )
    parser.add_argument(
        "--rank-source",
        choices=["teacher", "full", "mixed"],
        default="teacher",
        help="Candidate distribution for rank loss.",
    )
    parser.add_argument(
        "--rank-full-weight",
        type=float,
        default=0.25,
        help="When rank-source=mixed, blend this much full-image probability into teacher candidates.",
    )
    parser.add_argument("--seed", "--random-seed", dest="seed", type=int, default=364)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hflip-tta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-gpu-memory-fraction", type=float, default=0.85)
    parser.add_argument("--score-mode", default="max_margin", choices=["class", "margin", "max_margin", "entropy"])
    parser.add_argument("--blend-weights", default="0,0.15,0.25,0.35,0.50,0.65")
    parser.add_argument("--eval-train-every", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--submission-name", default=None)
    parser.add_argument(
        "--fit-on-validation",
        action="store_true",
        help="Train the crop head on train+validation after hyperparameters are fixed. Validation metrics are reported only as diagnostics.",
    )
    parser.add_argument(
        "--fixed-blend-weight",
        type=float,
        default=None,
        help="Override the saved best blend weight for final-fit submissions.",
    )
    parser.add_argument("--no-channels-last", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        if args.max_gpu_memory_fraction <= 0 or args.max_gpu_memory_fraction > 1:
            raise ValueError("--max-gpu-memory-fraction must be in (0, 1].")
        if args.max_gpu_memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction, cuda_index)

    base_model, checkpoint, image_size = load_base_model(checkpoint_path, device)
    use_channels_last = device.type == "cuda" and not args.no_channels_last
    if use_channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    base_params = count_parameters(base_model)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    teacher = None
    teacher_crop_size = args.teacher_crop_size
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        if teacher_checkpoint.get("idx_to_label") != checkpoint["idx_to_label"]:
            raise RuntimeError("Teacher and base label maps do not match.")
        teacher_model_name = args.teacher_model_name or teacher_checkpoint.get("args", {}).get(
            "model_name",
            "timm:tf_efficientnet_b4.ns_jft_in1k",
        )
        teacher_crop_size = teacher_crop_size or int(teacher_checkpoint.get("args", {}).get("image_size", 448))
        teacher = build_teacher(teacher_model_name, len(checkpoint["idx_to_label"]), pretrained=False).to(device)
        teacher.load_state_dict(teacher_checkpoint["model"])
        if use_channels_last:
            teacher = teacher.to(memory_format=torch.channels_last)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    train_cache = collect_crop_features(
        data_dir=data_dir,
        split="train",
        checkpoint_path=checkpoint_path,
        model=base_model,
        checkpoint=checkpoint,
        image_size=image_size,
        crop_size=args.crop_size,
        crop_padding=args.crop_padding,
        teacher=teacher,
        teacher_crop_size=int(teacher_crop_size or args.crop_size),
        teacher_hflip=args.teacher_hflip_tta,
        teacher_blend_detector_weight=args.teacher_blend_detector_weight,
        output_dir=output_dir,
        batch_size=args.predict_batch_size,
        device=device,
        hflip=args.hflip_tta,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        channels_last=use_channels_last,
        max_records=args.max_records,
    )
    val_cache = collect_crop_features(
        data_dir=data_dir,
        split="validation",
        checkpoint_path=checkpoint_path,
        model=base_model,
        checkpoint=checkpoint,
        image_size=image_size,
        crop_size=args.crop_size,
        crop_padding=args.crop_padding,
        teacher=teacher,
        teacher_crop_size=int(teacher_crop_size or args.crop_size),
        teacher_hflip=args.teacher_hflip_tta,
        teacher_blend_detector_weight=args.teacher_blend_detector_weight,
        output_dir=output_dir,
        batch_size=args.predict_batch_size,
        device=device,
        hflip=args.hflip_tta,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        channels_last=use_channels_last,
        max_records=args.max_records,
    )
    if args.fit_on_validation:
        train_cache = merge_feature_caches([train_cache, val_cache], split_name="trainval")
        print(
            json.dumps(
                {
                    "fit_on_validation": True,
                    "trainval_records": int(train_cache["labels"].numel()),
                    "note": "Validation labels are included only after hyperparameters are fixed; do not use these metrics for model selection.",
                },
                indent=2,
            )
        )

    feature_dim = int(train_cache["features"].shape[1])
    head = CropFeatureHead(
        feature_dim=feature_dim,
        num_classes=len(checkpoint["idx_to_label"]),
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    ).to(device)
    head_params = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    total_params = base_params + head_params
    if total_params > 10_000_000:
        raise RuntimeError(f"Base + crop head has {total_params:,} parameters, exceeding the 10M limit.")
    print(f"Base params: {base_params:,}, crop head params: {head_params:,}, combined final params: {total_params:,}")

    train_features = train_cache["features"]
    train_labels = train_cache["labels"]
    train_teacher_probs = train_cache.get("teacher_probs")
    if train_teacher_probs is not None and args.distill_weight <= 0:
        args.distill_weight = 0.55
    if args.rank_source == "teacher" and train_teacher_probs is None and args.rank_loss_weight > 0:
        raise RuntimeError("--rank-source teacher requires --teacher-checkpoint so teacher_probs are cached.")
    if args.rank_source == "teacher":
        rank_candidate_probs = train_teacher_probs
    elif args.rank_source == "full":
        rank_candidate_probs = train_cache["full_probs"]
    else:
        if train_teacher_probs is None:
            raise RuntimeError("--rank-source mixed requires --teacher-checkpoint so teacher_probs are cached.")
        full_weight = max(0.0, min(1.0, float(args.rank_full_weight)))
        rank_candidate_probs = full_weight * train_cache["full_probs"].float() + (1.0 - full_weight) * train_teacher_probs.float()
    train_indices = torch.arange(train_labels.numel(), dtype=torch.long)
    dataset = TensorDataset(train_features, train_labels, train_indices)
    class_counts = Counter(int(label) for label in train_labels.tolist())
    sample_weights = torch.as_tensor([1.0 / class_counts[int(label)] for label in train_labels.tolist()], dtype=torch.double)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    class_weight = torch.bincount(train_labels, minlength=len(checkpoint["idx_to_label"])).float()
    class_weight = (class_weight.sum() / class_weight.clamp_min(1.0)).sqrt()
    class_weight = (class_weight / class_weight.mean()).to(device)
    blend_weights = [float(item) for item in args.blend_weights.split(",") if item.strip()]

    base_train = evaluate_head(
        head=head,
        cache=train_cache,
        data_dir=data_dir,
        device=device,
        blend_weights=blend_weights,
        score_mode=args.score_mode,
    )
    base_val = evaluate_head(
        head=head,
        cache=val_cache,
        data_dir=data_dir,
        device=device,
        blend_weights=blend_weights,
        score_mode=args.score_mode,
    )
    print(json.dumps({"initial_train": base_train, "initial_val": base_val}, indent=2))

    best_ap = -1.0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        for batch_features, batch_labels, batch_indices in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            hard_loss = nn.functional.cross_entropy(
                logits,
                batch_labels,
                weight=class_weight,
                label_smoothing=args.label_smoothing,
            )
            loss = hard_loss
            if train_teacher_probs is not None and args.distill_weight > 0:
                temperature = max(args.distill_temperature, 1e-3)
                targets = train_teacher_probs[batch_indices].to(device).clamp_min(1e-8)
                kd_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / temperature, dim=1),
                    targets,
                    reduction="batchmean",
                ) * (temperature * temperature)
                weight = max(0.0, min(1.0, args.distill_weight))
                loss = (1.0 - weight) * hard_loss + weight * kd_loss
            if args.rank_loss_weight > 0:
                if rank_candidate_probs is None:
                    raise RuntimeError("Rank loss requested but no candidate probabilities are available.")
                rank_ce, rank_margin = candidate_rank_loss(
                    logits,
                    batch_labels,
                    rank_candidate_probs[batch_indices].to(device).float(),
                    topk=args.rank_topk,
                    margin=args.rank_margin,
                )
                loss = loss + float(args.rank_loss_weight) * (rank_ce + rank_margin)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        train_eval = None
        if args.eval_train_every > 0 and epoch % args.eval_train_every == 0:
            train_eval = evaluate_head(
                head=head,
                cache=train_cache,
                data_dir=data_dir,
                device=device,
                blend_weights=blend_weights,
                score_mode=args.score_mode,
            )
        val_eval = evaluate_head(
            head=head,
            cache=val_cache,
            data_dir=data_dir,
            device=device,
            blend_weights=blend_weights,
            score_mode=args.score_mode,
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "validation": val_eval,
        }
        if train_eval is not None:
            row["train"] = train_eval
        history.append(row)
        print(json.dumps(row, indent=2))
        checkpoint_out = {
            "model": head.state_dict(),
            "base_checkpoint": str(checkpoint_path),
            "base_params": base_params,
            "head_params": head_params,
            "combined_params": total_params,
            "feature_dim": feature_dim,
            "idx_to_label": checkpoint["idx_to_label"],
            "label_to_idx": checkpoint["label_to_idx"],
            "args": vars(args),
            "metrics": val_eval,
        }
        torch.save(checkpoint_out, output_dir / "last.pt")
        if val_eval["best_mAP50"] > best_ap:
            best_ap = val_eval["best_mAP50"]
            torch.save(checkpoint_out, output_dir / "best.pt")

    (output_dir / "training_summary.json").write_text(
        json.dumps({"history": history, "best_mAP50": best_ap, "combined_params": total_params}, indent=2),
        encoding="utf-8",
    )

    if args.submission_name:
        if args.fixed_blend_weight is not None:
            best_path = output_dir / "best.pt"
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            best_payload.setdefault("metrics", {})["best_blend_weight"] = float(args.fixed_blend_weight)
            torch.save(best_payload, best_path)
        write_prediction_csv(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            head_path=output_dir / "best.pt",
            output_csv=Path("submissions_to_test") / args.submission_name,
            batch_size=args.predict_batch_size,
            device=device,
            hflip=args.hflip_tta,
            score_mode=args.score_mode,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            channels_last=use_channels_last,
        )


if __name__ == "__main__":
    main()
