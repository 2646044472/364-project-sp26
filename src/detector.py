from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from torchvision.ops import generalized_box_iou_loss


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PARENTHETICAL_LABEL_PATTERN = re.compile(r"\s*\([^)]*\)")
HF_BIRDS_B2_MODEL_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"
TIMM_BACKBONES = {
    "timm_tf_efficientnet_b0_ns": ("tf_efficientnet_b0.ns_jft_in1k", 1280),
    "timm_tf_efficientnet_b1_ns": ("tf_efficientnet_b1.ns_jft_in1k", 1280),
    "timm_tf_efficientnet_b2_ns": ("tf_efficientnet_b2.ns_jft_in1k", 1408),
    "timm_convnext_femto": ("convnext_femto", 384),
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_loader_kwargs(
    num_workers: int,
    prefetch_factor: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    if pin_memory is None:
        pin_memory = torch.device(device).type == "cuda"
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_timm_features(model_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Install timm to use timm_* backbones: pip install timm") from exc
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(-1,))
    channels = int(model.feature_info.channels()[-1])
    return model, channels


def run_feature_backbone(features: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = features(images)
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


def simplify_species_label(label: str) -> str:
    simplified = PARENTHETICAL_LABEL_PATTERN.sub("", label)
    return " ".join(simplified.split())


def find_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images[path.name.lower()] = path
        images[path.stem.lower()] = path
    return images


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _parse_prediction_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return parsed if isinstance(parsed, dict) else None


def _object_value(obj: dict[str, Any], names: list[str]) -> Any:
    lower_to_key = {str(key).lower(): key for key in obj}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None:
            return obj[key]
    return None


def _row_value(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        for column in row.index:
            if column.lower() == name.lower():
                return row[column]
    return None


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_box(box: list[float], width: int, height: int) -> list[float]:
    x_min, y_min, x_max, y_max = box
    if max(abs(v) for v in box) > 1.5:
        x_min /= max(width, 1)
        x_max /= max(width, 1)
        y_min /= max(height, 1)
        y_max /= max(height, 1)
    x_min, x_max = sorted((max(0.0, min(1.0, x_min)), max(0.0, min(1.0, x_max))))
    y_min, y_max = sorted((max(0.0, min(1.0, y_min)), max(0.0, min(1.0, y_max))))
    return [x_min, y_min, x_max, y_max]


def _crop_around_box(
    image: Image.Image,
    box: list[float],
    padding: float = 0.18,
    jitter: float = 0.12,
) -> tuple[Image.Image, list[float]]:
    width, height = image.size
    x_min, y_min, x_max, y_max = box
    box_width = max(1e-4, x_max - x_min)
    box_height = max(1e-4, y_max - y_min)
    pad_x = box_width * random.uniform(max(0.0, padding - jitter), padding + jitter)
    pad_y = box_height * random.uniform(max(0.0, padding - jitter), padding + jitter)
    crop = [
        max(0.0, x_min - pad_x),
        max(0.0, y_min - pad_y),
        min(1.0, x_max + pad_x),
        min(1.0, y_max + pad_y),
    ]
    left = int(math.floor(crop[0] * width))
    top = int(math.floor(crop[1] * height))
    right = int(math.ceil(crop[2] * width))
    bottom = int(math.ceil(crop[3] * height))
    if right <= left + 1 or bottom <= top + 1:
        return image, box
    cropped = image.crop((left, top, right, bottom))
    crop_width = max(right - left, 1)
    crop_height = max(bottom - top, 1)
    adjusted = [
        (x_min * width - left) / crop_width,
        (y_min * height - top) / crop_height,
        (x_max * width - left) / crop_width,
        (y_max * height - top) / crop_height,
    ]
    return cropped, _normalize_box(adjusted, 1, 1)


@dataclass
class BirdRecord:
    image_id: str
    image_path: Path
    label: str | None = None
    box: list[float] | None = None


def _read_split_ids(path: Path) -> set[str]:
    return {line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values


def _load_birdwatching_txt_records(root: Path, split: str) -> list[BirdRecord] | None:
    images_txt_candidates = sorted(root.rglob("images.txt"))
    if not images_txt_candidates:
        return None
    base = images_txt_candidates[0].parent
    split_file = base / f"{split}_split.txt"
    if split == "validation" and not split_file.exists():
        split_file = base / "val_split.txt"
    if split == "val" and not split_file.exists():
        split_file = base / "val_split.txt"
    if not split_file.exists():
        return None

    split_ids = _read_split_ids(split_file)
    image_relpaths = _read_key_value_file(base / "images.txt")
    class_ids = _read_key_value_file(base / "image_class_labels.txt")
    class_names = _read_key_value_file(base / "classes.txt") if (base / "classes.txt").exists() else {}
    raw_boxes = _read_key_value_file(base / "bounding_boxes.txt") if (base / "bounding_boxes.txt").exists() else {}
    image_root = base / "images"

    records: list[BirdRecord] = []
    for image_id in sorted(split_ids):
        relpath = image_relpaths.get(image_id)
        if relpath is None:
            continue
        image_path = image_root / relpath
        if not image_path.exists():
            continue
        label: str | None = None
        class_id = class_ids.get(image_id)
        if class_id is not None:
            label = class_names.get(class_id, class_id)
        box: list[float] | None = None
        raw_box = raw_boxes.get(image_id)
        if raw_box is not None:
            values = [float(value) for value in raw_box.split()[:4]]
            with Image.open(image_path) as image:
                box = _normalize_box(values, image.width, image.height)
        records.append(BirdRecord(image_id=image_id, image_path=image_path, label=label, box=box))
    return records


def load_records(root: Path, split: str) -> list[BirdRecord]:
    split = split.lower()
    txt_records = _load_birdwatching_txt_records(root, split)
    if txt_records is not None:
        if split not in {"test"}:
            txt_records = [record for record in txt_records if record.label is not None and record.box is not None]
        if txt_records:
            return txt_records

    image_index = find_images(root)
    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {root}")

    preferred = [path for path in csv_paths if split in path.stem.lower()]
    if split == "test":
        preferred += [path for path in csv_paths if "sample" in path.stem.lower()]
    if split == "test" and preferred:
        candidates = preferred
    else:
        candidates = preferred + [path for path in csv_paths if path not in preferred]

    best_records: list[BirdRecord] = []
    for csv_path in candidates:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        id_col = _first_existing(
            list(df.columns),
            ["id", "Id", "image_id", "image", "filename", "file_name", "path"],
        )
        if id_col is None:
            continue
        label_col = _first_existing(
            list(df.columns),
            ["label", "species", "class", "class_name", "category", "name"],
        )
        prediction_col = _first_existing(list(df.columns), ["Predictions", "prediction", "annotations"])
        records: list[BirdRecord] = []
        for _, row in df.iterrows():
            image_id = str(row[id_col])
            key = Path(image_id).name.lower()
            image_path = image_index.get(key) or image_index.get(Path(key).stem)
            if image_path is None:
                continue
            label = str(row[label_col]) if label_col and not pd.isna(row[label_col]) else None
            box: list[float] | None = None
            if prediction_col:
                obj = _parse_prediction_object(row[prediction_col])
                if obj:
                    label = label or str(obj.get("label", ""))
                    values = [
                        _to_float(_object_value(obj, ["x_min", "xmin"])),
                        _to_float(_object_value(obj, ["y_min", "ymin"])),
                        _to_float(_object_value(obj, ["x_max", "xmax"])),
                        _to_float(_object_value(obj, ["y_max", "ymax"])),
                    ]
                    if all(value is not None for value in values):
                        box = [float(value) for value in values if value is not None]
            if box is None:
                values = [
                    _to_float(_row_value(row, ["x_min", "xmin", "left", "x1"])),
                    _to_float(_row_value(row, ["y_min", "ymin", "top", "y1"])),
                    _to_float(_row_value(row, ["x_max", "xmax", "right", "x2"])),
                    _to_float(_row_value(row, ["y_max", "ymax", "bottom", "y2"])),
                ]
                if all(value is not None for value in values):
                    box = [float(value) for value in values if value is not None]
                else:
                    xywh = [
                        _to_float(_row_value(row, ["x", "x_center", "cx"])),
                        _to_float(_row_value(row, ["y", "y_center", "cy"])),
                        _to_float(_row_value(row, ["width", "w"])),
                        _to_float(_row_value(row, ["height", "h"])),
                    ]
                    if all(value is not None for value in xywh):
                        x, y, w, h = [float(value) for value in xywh if value is not None]
                        box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            if box is not None:
                with Image.open(image_path) as image:
                    box = _normalize_box(box, image.width, image.height)
            records.append(BirdRecord(image_id=image_id, image_path=image_path, label=label, box=box))
        if len(records) > len(best_records):
            best_records = records

    if not best_records:
        raise RuntimeError(f"Could not build {split} records from {root}")
    if split != "test":
        best_records = [record for record in best_records if record.label is not None and record.box is not None]
    return best_records


def build_label_maps(records: list[BirdRecord]) -> tuple[dict[str, int], list[str]]:
    labels = sorted({record.label for record in records if record.label is not None})
    label_to_idx = {label: index for index, label in enumerate(labels)}
    return label_to_idx, labels


def load_class_id_labels(root: Path) -> dict[int, str]:
    classes_candidates = sorted(root.rglob("classes.txt"))
    if not classes_candidates:
        raise FileNotFoundError(f"Could not find classes.txt under {root}")
    labels: dict[int, str] = {}
    for line in classes_candidates[0].read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1]
    if not labels:
        raise RuntimeError(f"No class labels found in {classes_candidates[0]}")
    return labels


def build_prediction_label_map(root: Path, idx_to_label: list[str]) -> list[str]:
    """Map internal class indices back to the raw training label ids used by submissions."""
    class_id_to_label = load_class_id_labels(root)
    label_to_class_id = {label: str(class_id) for class_id, label in class_id_to_label.items()}
    normalized_to_class_id = {
        " ".join(label.split()).lower(): str(class_id)
        for class_id, label in class_id_to_label.items()
    }
    output_labels: list[str] = []
    missing: list[str] = []
    for label in idx_to_label:
        if label in label_to_class_id:
            output_labels.append(label_to_class_id[label])
            continue
        normalized = " ".join(str(label).split()).lower()
        if normalized in normalized_to_class_id:
            output_labels.append(normalized_to_class_id[normalized])
            continue
        if str(label).isdigit():
            output_labels.append(str(label))
            continue
        missing.append(str(label))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Could not map checkpoint labels to raw class ids: {preview}")
    return output_labels


def folder_label(record: BirdRecord, class_id_to_label: dict[int, str]) -> str:
    folder = record.image_path.parent.name
    if not folder.isdigit():
        raise ValueError(f"Image path does not contain a numeric class folder: {record.image_path}")
    class_id = int(folder)
    if class_id not in class_id_to_label:
        raise KeyError(f"Class id {class_id} from {record.image_path} is missing from classes.txt")
    return class_id_to_label[class_id]


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    return torch.stack(
        [
            (x_min + x_max) * 0.5,
            (y_min + y_max) * 0.5,
            (x_max - x_min).clamp_min(1e-4),
            (y_max - y_min).clamp_min(1e-4),
        ],
        dim=-1,
    )


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    x_min = cx - width * 0.5
    y_min = cy - height * 0.5
    x_max = cx + width * 0.5
    y_max = cy + height * 0.5
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1).clamp(0.0, 1.0)


def pad_boxes(boxes: torch.Tensor, padding: float = 0.08) -> torch.Tensor:
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    width = (x_max - x_min).clamp_min(1e-4)
    height = (y_max - y_min).clamp_min(1e-4)
    padded = torch.stack(
        [
            x_min - width * padding,
            y_min - height * padding,
            x_max + width * padding,
            y_max + height * padding,
        ],
        dim=-1,
    )
    return padded.clamp(0.0, 1.0)


def jitter_boxes(
    boxes: torch.Tensor,
    padding_min: float = 0.05,
    padding_max: float = 0.30,
    center_jitter: float = 0.12,
) -> torch.Tensor:
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    width = (x_max - x_min).clamp_min(1e-4)
    height = (y_max - y_min).clamp_min(1e-4)
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    padding = torch.empty_like(width).uniform_(padding_min, padding_max)
    offset_x = torch.empty_like(width).uniform_(-center_jitter, center_jitter) * width
    offset_y = torch.empty_like(height).uniform_(-center_jitter, center_jitter) * height
    new_width = width * (1.0 + 2.0 * padding)
    new_height = height * (1.0 + 2.0 * padding)
    center_x = center_x + offset_x
    center_y = center_y + offset_y
    jittered = torch.stack(
        [
            center_x - new_width * 0.5,
            center_y - new_height * 0.5,
            center_x + new_width * 0.5,
            center_y + new_height * 0.5,
        ],
        dim=-1,
    )
    return jittered.clamp(0.0, 1.0)


def roi_align_single(images: torch.Tensor, boxes_xyxy: torch.Tensor, output_size: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    theta = images.new_zeros((batch, 2, 3))
    x_min, y_min, x_max, y_max = boxes_xyxy.unbind(dim=-1)
    theta[:, 0, 0] = (x_max - x_min).clamp_min(1e-4)
    theta[:, 1, 1] = (y_max - y_min).clamp_min(1e-4)
    theta[:, 0, 2] = x_min + x_max - 1.0
    theta[:, 1, 2] = y_min + y_max - 1.0
    grid = F.affine_grid(theta, size=(batch, channels, output_size, output_size), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def horizontal_flip_boxes(boxes: torch.Tensor) -> torch.Tensor:
    flipped = boxes.clone()
    flipped[:, 0] = 1.0 - boxes[:, 2]
    flipped[:, 2] = 1.0 - boxes[:, 0]
    return flipped


def _predict_full_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    hflip: bool = False,
    logit_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(images)
    logits = outputs["class_logits"].float()
    if logit_bias is not None:
        logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
    probs = logits.softmax(dim=1)
    boxes = outputs["box_xyxy"].float()
    if not hflip:
        return probs, boxes

    flipped_outputs = model(torch.flip(images, dims=[3]))
    flipped_logits = flipped_outputs["class_logits"].float()
    if logit_bias is not None:
        flipped_logits = flipped_logits + logit_bias.to(device=flipped_logits.device, dtype=flipped_logits.dtype)
    flipped_probs = flipped_logits.softmax(dim=1)
    flipped_boxes = horizontal_flip_boxes(flipped_outputs["box_xyxy"].float())
    return (probs + flipped_probs) * 0.5, (boxes + flipped_boxes) * 0.5


def confidence_from_probs(probs: torch.Tensor, mode: str = "max") -> tuple[torch.Tensor, torch.Tensor]:
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    scores = top2.values[:, 0]
    labels = top2.indices[:, 0]
    if mode == "max":
        confidence = scores
    elif mode == "margin":
        second = top2.values[:, 1] if top2.values.shape[1] > 1 else torch.zeros_like(scores)
        confidence = (scores - second).clamp_min(0.0)
    elif mode == "max_margin":
        second = top2.values[:, 1] if top2.values.shape[1] > 1 else torch.zeros_like(scores)
        confidence = (scores * (scores - second).clamp_min(0.0)).sqrt()
    elif mode == "entropy":
        entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=1)
        confidence = 1.0 - entropy / math.log(max(probs.shape[1], 2))
    else:
        raise ValueError(f"Unsupported confidence mode: {mode}")
    return confidence, labels


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    hflip: bool = False,
    crop_classify: bool = False,
    crop_padding: float = 0.10,
    crop_size: int | None = None,
    crop_full_weight: float = 0.20,
    logit_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs, boxes = _predict_full_with_tta(model, images, hflip=hflip, logit_bias=logit_bias)
    if not crop_classify:
        return probs, boxes

    output_size = crop_size or int(images.shape[-1])
    crop_boxes = pad_boxes(boxes.detach(), padding=crop_padding)
    crops = roi_align_single(images, crop_boxes, output_size)
    crop_probs, _ = _predict_full_with_tta(model, crops, hflip=hflip, logit_bias=logit_bias)
    full_weight = max(0.0, min(1.0, crop_full_weight))
    return full_weight * probs + (1.0 - full_weight) * crop_probs, boxes


def _resize_letterbox(image: Image.Image, box: list[float] | None, size: int) -> tuple[Image.Image, list[float] | None]:
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    if box is None:
        return canvas, None
    x_min, y_min, x_max, y_max = box
    adjusted = [
        (x_min * width * scale + pad_x) / size,
        (y_min * height * scale + pad_y) / size,
        (x_max * width * scale + pad_x) / size,
        (y_max * height * scale + pad_y) / size,
    ]
    return canvas, _normalize_box(adjusted, 1, 1)


def unletterbox_box(box: np.ndarray, image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        width, height = image.size
    scale = min(image_size / width, image_size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    pad_x = (image_size - new_width) // 2
    pad_y = (image_size - new_height) // 2
    x_min, y_min, x_max, y_max = [float(value) for value in box]
    original_box = [
        (x_min * image_size - pad_x) / max(scale * width, 1e-6),
        (y_min * image_size - pad_y) / max(scale * height, 1e-6),
        (x_max * image_size - pad_x) / max(scale * width, 1e-6),
        (y_max * image_size - pad_y) / max(scale * height, 1e-6),
    ]
    return np.asarray(_normalize_box(original_box, 1, 1), dtype=np.float32)


def _augment(image: Image.Image, box: list[float]) -> tuple[Image.Image, list[float]]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x_min, y_min, x_max, y_max = box
        box = [1.0 - x_max, y_min, 1.0 - x_min, y_max]
    if random.random() < 0.8:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    return image, box


def extract_species_name(label: str) -> str:
    return simplify_species_label(label).strip()


def build_species_maps(labels: list[str]) -> tuple[dict[str, int], list[str], list[int]]:
    species_names = sorted({extract_species_name(label) for label in labels})
    species_to_idx = {name: index for index, name in enumerate(species_names)}
    class_to_species = [species_to_idx[extract_species_name(label)] for label in labels]
    return species_to_idx, species_names, class_to_species


class BirdDataset(Dataset):
    def __init__(
        self,
        records: list[BirdRecord],
        label_to_idx: dict[str, int] | None,
        image_size: int,
        training: bool,
        class_to_species: list[int] | None = None,
        bbox_crop_prob: float = 0.0,
    ) -> None:
        self.records = records
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.training = training
        self.class_to_species = class_to_species
        self.bbox_crop_prob = bbox_crop_prob
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        box = record.box.copy() if record.box is not None else None
        if self.training and box is not None and self.bbox_crop_prob > 0 and random.random() < self.bbox_crop_prob:
            image, box = _crop_around_box(image, box)
        image, box = _resize_letterbox(image, box, self.image_size)
        if self.training and box is not None:
            image, box = _augment(image, box)

        array = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        item: dict[str, Any] = {"image": tensor, "image_id": record.image_id}
        if self.label_to_idx is not None and record.label is not None and box is not None:
            label_idx = self.label_to_idx[record.label]
            item["label"] = torch.tensor(label_idx, dtype=torch.long)
            item["box_xyxy"] = torch.tensor(box, dtype=torch.float32)
            item["box_cxcywh"] = box_xyxy_to_cxcywh(item["box_xyxy"])
            if self.class_to_species is not None:
                item["species_label"] = torch.tensor(self.class_to_species[label_idx], dtype=torch.long)
        return item


class BirdDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone_name: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            channels = 576
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b1(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b2(weights=weights)
            channels = 1408
        elif backbone_name in TIMM_BACKBONES:
            timm_name, _expected_channels = TIMM_BACKBONES[backbone_name]
            backbone_features, channels = _build_timm_features(timm_name, pretrained=pretrained)
            backbone = None
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.features = backbone_features if backbone_name in TIMM_BACKBONES else backbone.features
        self.backbone_name = backbone_name
        self.feature_channels = channels
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels + 4),
            nn.Linear(channels + 4, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(channels + 4),
            nn.Linear(channels + 4, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def _forward_once(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = run_feature_backbone(self.features, images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        vector = torch.cat([pooled, center, spread], dim=1)

        class_logits = self.classifier(vector)
        box_cxcywh = torch.sigmoid(self.box_head(vector))
        box_cxcywh = torch.cat([box_cxcywh[:, :2], box_cxcywh[:, 2:].clamp(0.02, 1.0)], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
        return {
            "class_logits": class_logits,
            "box_cxcywh": box_cxcywh,
            "box_xyxy": box_xyxy,
            "pooled_vector": vector,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._forward_once(images)
        outputs.pop("pooled_vector", None)
        return outputs


class CropConsistencyDetector(BirdDetector):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone_name: str = "timm_tf_efficientnet_b2_ns",
        crop_size: int = 320,
        crop_padding: float = 0.18,
        crop_full_logit_weight: float = 0.40,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained, dropout=dropout, backbone_name=backbone_name)
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.crop_full_logit_weight = crop_full_logit_weight

    def forward(self, images: torch.Tensor, crop_boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        full_outputs = self._forward_once(images)
        if crop_boxes is None:
            crop_source = pad_boxes(full_outputs["box_xyxy"].detach(), padding=self.crop_padding)
        else:
            crop_source = crop_boxes
        crops = roi_align_single(images, crop_source, self.crop_size)
        crop_outputs = self._forward_once(crops)
        full_weight = max(0.0, min(1.0, self.crop_full_logit_weight))
        class_logits = full_weight * full_outputs["class_logits"] + (1.0 - full_weight) * crop_outputs["class_logits"]
        return {
            "class_logits": class_logits,
            "full_class_logits": full_outputs["class_logits"],
            "crop_class_logits": crop_outputs["class_logits"],
            "box_cxcywh": full_outputs["box_cxcywh"],
            "box_xyxy": full_outputs["box_xyxy"],
        }


class BilinearAttentionDetector(BirdDetector):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone_name: str = "efficientnet_b2",
        bilinear_dim: int = 224,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained, dropout=dropout, backbone_name=backbone_name)
        if backbone_name in {"efficientnet_b0", "efficientnet_b1"}:
            channels = 1280
        elif backbone_name == "efficientnet_b2":
            channels = 1408
        elif backbone_name == "mobilenet_v3_large":
            channels = 960
        elif backbone_name == "mobilenet_v3_small":
            channels = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.bilinear_a = nn.Conv2d(channels, bilinear_dim, kernel_size=1)
        self.bilinear_b = nn.Conv2d(channels, bilinear_dim, kernel_size=1)
        self.bilinear_classifier = nn.Sequential(
            nn.LayerNorm(bilinear_dim + 4),
            nn.Dropout(dropout),
            nn.Linear(bilinear_dim + 4, num_classes),
        )
        self.logit_mix = nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        vector = torch.cat([pooled, center, spread], dim=1)

        base_logits = self.classifier(vector)
        box_cxcywh = torch.sigmoid(self.box_head(vector))
        box_cxcywh = torch.cat([box_cxcywh[:, :2], box_cxcywh[:, 2:].clamp(0.02, 1.0)], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)

        proj_a = self.bilinear_a(feats)
        proj_b = self.bilinear_b(feats)
        bilinear = (proj_a * proj_b).flatten(2)
        bilinear = (bilinear * weights).sum(dim=-1)
        bilinear = torch.sign(bilinear) * torch.sqrt(bilinear.abs() + 1e-6)
        bilinear = F.normalize(bilinear, dim=1)
        bilinear_vector = torch.cat([bilinear, center, spread], dim=1)
        bilinear_logits = self.bilinear_classifier(bilinear_vector)
        mix = torch.sigmoid(self.logit_mix)
        class_logits = mix * base_logits + (1.0 - mix) * bilinear_logits
        return {
            "class_logits": class_logits,
            "base_class_logits": base_logits,
            "bilinear_class_logits": bilinear_logits,
            "box_cxcywh": box_cxcywh,
            "box_xyxy": box_xyxy,
        }


class MultiAttentionPartDetector(BirdDetector):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone_name: str = "efficientnet_b2",
        num_parts: int = 3,
        part_dim: int = 64,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained, dropout=dropout, backbone_name=backbone_name)
        channels = self.feature_channels
        self.num_parts = num_parts
        self.part_attention = nn.Conv2d(channels, num_parts, kernel_size=1, bias=False)
        self.part_project = nn.Linear(channels, part_dim, bias=False)
        fusion_dim = channels + num_parts * part_dim + 4
        self.part_classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 352),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(352, num_classes),
        )
        self.logit_mix = nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = run_feature_backbone(self.features, images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        vector = torch.cat([pooled, center, spread], dim=1)

        base_logits = self.classifier(vector)
        box_cxcywh = torch.sigmoid(self.box_head(vector))
        box_cxcywh = torch.cat([box_cxcywh[:, :2], box_cxcywh[:, 2:].clamp(0.02, 1.0)], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)

        part_maps = self.part_attention(feats).flatten(2)
        part_weights = torch.softmax(part_maps, dim=-1)
        feat_grid = feats.flatten(2).transpose(1, 2)
        part_vectors = torch.einsum("bpn,bnc->bpc", part_weights, feat_grid)
        part_vectors = self.part_project(part_vectors)
        part_vectors = torch.sign(part_vectors) * torch.sqrt(part_vectors.abs() + 1e-6)
        part_vectors = F.normalize(part_vectors.flatten(1), dim=1)
        fused = torch.cat([pooled, part_vectors, center, spread], dim=1)
        part_logits = self.part_classifier(fused)
        mix = torch.sigmoid(self.logit_mix)
        class_logits = mix * base_logits + (1.0 - mix) * part_logits
        return {
            "class_logits": class_logits,
            "base_class_logits": base_logits,
            "part_class_logits": part_logits,
            "box_cxcywh": box_cxcywh,
            "box_xyxy": box_xyxy,
        }


class MultiStageResidualDetector(BirdDetector):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        backbone_name: str = "efficientnet_b2",
        stage_dim: int = 64,
        residual_hidden: int = 256,
    ) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained, dropout=dropout, backbone_name=backbone_name)
        if backbone_name == "efficientnet_b0":
            stage_channels = [80, 112, 320]
            self.stage_indices = [4, 5, 7]
        elif backbone_name == "efficientnet_b1":
            stage_channels = [80, 112, 320]
            self.stage_indices = [4, 5, 7]
        elif backbone_name == "efficientnet_b2":
            stage_channels = [88, 120, 352]
            self.stage_indices = [4, 5, 7]
        else:
            raise ValueError(f"Unsupported multistage backbone: {backbone_name}")
        self.stage_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, stage_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(stage_dim),
                    nn.Hardswish(),
                )
                for in_channels in stage_channels
            ]
        )
        stage_vector_dim = stage_dim * len(stage_channels)
        self.stage_classifier = nn.Sequential(
            nn.LayerNorm(stage_vector_dim),
            nn.Linear(stage_vector_dim, residual_hidden),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, num_classes),
        )
        self.stage_logit_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        stage_vectors: list[torch.Tensor] = []
        stage_lookup = dict(zip(self.stage_indices, self.stage_projectors))
        feats = images
        for index, layer in enumerate(self.features):
            feats = layer(feats)
            projector = stage_lookup.get(index)
            if projector is not None:
                stage_vectors.append(F.adaptive_avg_pool2d(projector(feats), 1).flatten(1))

        batch, _channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        vector = torch.cat([pooled, center, spread], dim=1)

        base_logits = self.classifier(vector)
        stage_logits = self.stage_classifier(torch.cat(stage_vectors, dim=1))
        class_logits = base_logits + torch.sigmoid(self.stage_logit_scale) * stage_logits
        box_cxcywh = torch.sigmoid(self.box_head(vector))
        box_cxcywh = torch.cat([box_cxcywh[:, :2], box_cxcywh[:, 2:].clamp(0.02, 1.0)], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
        return {
            "class_logits": class_logits,
            "base_class_logits": base_logits,
            "stage_class_logits": stage_logits,
            "box_cxcywh": box_cxcywh,
            "box_xyxy": box_xyxy,
        }


class BirdGridDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.15,
        backbone_name: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            channels = 576
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b1(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b2(weights=weights)
            channels = 1408
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        hidden = 256
        self.features = backbone.features
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, 1 + num_classes + 4, kernel_size=1),
        )
        self.global_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.features(images)
        pred = self.head(feats)
        objectness_logits = pred[:, 0]
        class_logits_map = pred[:, 1 : 1 + self.num_classes]
        box_raw = pred[:, 1 + self.num_classes :].permute(0, 2, 3, 1)
        batch, height, width = objectness_logits.shape

        flat_objectness = objectness_logits.flatten(1)
        weights = torch.softmax(flat_objectness, dim=1)
        best_index = flat_objectness.argmax(dim=1)
        row = best_index // width
        col = best_index % width

        selected_class_logits = class_logits_map.permute(0, 2, 3, 1)[torch.arange(batch, device=images.device), row, col]
        global_class_logits = self.global_classifier(feats)
        class_logits = 0.75 * global_class_logits + 0.25 * selected_class_logits
        selected_box = box_raw[torch.arange(batch, device=images.device), row, col]
        grid_x = (col.float() + torch.sigmoid(selected_box[:, 0])) / width
        grid_y = (row.float() + torch.sigmoid(selected_box[:, 1])) / height
        box_wh = torch.sigmoid(selected_box[:, 2:]).clamp(0.02, 1.0)
        box_cxcywh = torch.cat([grid_x[:, None], grid_y[:, None], box_wh], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
        return {
            "objectness_logits": objectness_logits,
            "class_logits_map": class_logits_map,
            "box_raw_map": box_raw,
            "global_class_logits": global_class_logits,
            "selected_class_logits": selected_class_logits,
            "class_logits": class_logits,
            "box_cxcywh": box_cxcywh,
            "box_xyxy": box_xyxy,
        }


class SharedCropDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        backbone_name: str = "efficientnet_b2",
        crop_size: int = 260,
        crop_padding: float = 0.10,
        hierarchy_num_classes: int | None = None,
    ) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
            channels = 960
        elif backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            channels = 576
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b1(weights=weights)
            channels = 1280
        elif backbone_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b2(weights=weights)
            channels = 1408
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.features = backbone.features
        self.backbone_name = backbone_name
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.hierarchy_num_classes = hierarchy_num_classes
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels + 4),
            nn.Linear(channels + 4, 384),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(channels + 4),
            nn.Linear(channels + 4, 192),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(192, 4),
        )
        self.hierarchy_classifier = (
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hierarchy_num_classes),
            )
            if hierarchy_num_classes is not None
            else None
        )

    def _pool(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.features(images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        vector = torch.cat([pooled, center, spread], dim=1)
        return pooled, vector, feats

    def forward(self, images: torch.Tensor, crop_boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pooled, vector, _ = self._pool(images)
        coarse_box_cxcywh = torch.sigmoid(self.box_head(vector))
        coarse_box_cxcywh = torch.cat([coarse_box_cxcywh[:, :2], coarse_box_cxcywh[:, 2:].clamp(0.04, 1.0)], dim=1)
        coarse_box_xyxy = box_cxcywh_to_xyxy(coarse_box_cxcywh)

        crop_source = crop_boxes if crop_boxes is not None else coarse_box_xyxy.detach()
        crop_box_xyxy = pad_boxes(crop_source, padding=self.crop_padding)
        crops = roi_align_single(images, crop_box_xyxy, self.crop_size)
        crop_pooled, crop_vector, _ = self._pool(crops)
        class_logits = self.classifier(crop_vector)

        crop_box_cxcywh = torch.sigmoid(self.box_head(crop_vector))
        crop_box_cxcywh = torch.cat([crop_box_cxcywh[:, :2], crop_box_cxcywh[:, 2:].clamp(0.04, 1.0)], dim=1)
        crop_relative_xyxy = box_cxcywh_to_xyxy(crop_box_cxcywh)
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box_xyxy.unbind(dim=-1)
        crop_w = (crop_x_max - crop_x_min).clamp_min(1e-4)
        crop_h = (crop_y_max - crop_y_min).clamp_min(1e-4)
        refined_box_xyxy = torch.stack(
            [
                crop_x_min + crop_relative_xyxy[:, 0] * crop_w,
                crop_y_min + crop_relative_xyxy[:, 1] * crop_h,
                crop_x_min + crop_relative_xyxy[:, 2] * crop_w,
                crop_y_min + crop_relative_xyxy[:, 3] * crop_h,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        hierarchy_logits = self.hierarchy_classifier(crop_pooled) if self.hierarchy_classifier is not None else None
        result = {
            "class_logits": class_logits,
            "box_cxcywh": box_xyxy_to_cxcywh(refined_box_xyxy),
            "box_xyxy": refined_box_xyxy,
            "coarse_box_xyxy": coarse_box_xyxy,
            "coarse_box_cxcywh": coarse_box_cxcywh,
        }
        if hierarchy_logits is not None:
            result["hierarchy_logits"] = hierarchy_logits
        return result


class FusionPrototypeCropDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.22,
        backbone_name: str = "timm_tf_efficientnet_b1_ns",
        crop_size: int = 320,
        crop_padding: float = 0.10,
        fusion_dim: int = 704,
        prototype_dim: int = 352,
        crop_full_logit_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if backbone_name == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
            self.features = backbone.features
            channels = 960
        elif backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            channels = 576
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            self.features = backbone.features
            channels = 1280
        elif backbone_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b1(weights=weights)
            self.features = backbone.features
            channels = 1280
        elif backbone_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b2(weights=weights)
            self.features = backbone.features
            channels = 1408
        elif backbone_name in TIMM_BACKBONES:
            timm_name, _expected_channels = TIMM_BACKBONES[backbone_name]
            self.features, channels = _build_timm_features(timm_name, pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.backbone_name = backbone_name
        self.feature_channels = channels
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.crop_full_logit_weight = crop_full_logit_weight
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        vector_dim = channels + 4
        fusion_input_dim = vector_dim * 4
        self.full_classifier = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.crop_classifier = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim, num_classes),
        )
        self.prototype_projection = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, prototype_dim),
        )
        self.prototype_weight = nn.Parameter(torch.empty(num_classes, prototype_dim))
        nn.init.trunc_normal_(self.prototype_weight, std=0.02)
        self.prototype_scale = nn.Parameter(torch.tensor(16.0))
        self.logit_mix = nn.Parameter(torch.tensor([0.15, 0.35, 0.35, 0.15]))
        self.box_head = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 4),
        )
        self.crop_box_head = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 4),
        )

    def _pool(self, images: torch.Tensor) -> torch.Tensor:
        feats = run_feature_backbone(self.features, images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)
        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()
        return torch.cat([pooled, center, spread], dim=1)

    def forward(self, images: torch.Tensor, crop_boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        full_vector = self._pool(images)
        coarse_box_cxcywh = torch.sigmoid(self.box_head(full_vector))
        coarse_box_cxcywh = torch.cat([coarse_box_cxcywh[:, :2], coarse_box_cxcywh[:, 2:].clamp(0.04, 1.0)], dim=1)
        coarse_box_xyxy = box_cxcywh_to_xyxy(coarse_box_cxcywh)

        crop_source = crop_boxes if crop_boxes is not None else coarse_box_xyxy.detach()
        crop_box_xyxy = pad_boxes(crop_source, padding=self.crop_padding)
        crops = roi_align_single(images, crop_box_xyxy, self.crop_size)
        crop_vector = self._pool(crops)

        fusion_vector = torch.cat(
            [full_vector, crop_vector, full_vector * crop_vector, (full_vector - crop_vector).abs()],
            dim=1,
        )
        full_logits = self.full_classifier(full_vector)
        crop_logits = self.crop_classifier(crop_vector)
        fusion_logits = self.fusion_head(fusion_vector)
        prototype_features = F.normalize(self.prototype_projection(fusion_vector), dim=1)
        prototype_weight = F.normalize(self.prototype_weight, dim=1)
        prototype_logits = prototype_features @ prototype_weight.t() * self.prototype_scale.clamp(4.0, 32.0)
        mix = torch.softmax(self.logit_mix, dim=0)
        class_logits = (
            mix[0] * full_logits
            + mix[1] * crop_logits
            + mix[2] * fusion_logits
            + mix[3] * prototype_logits
        )

        crop_box_cxcywh = torch.sigmoid(self.crop_box_head(crop_vector))
        crop_box_cxcywh = torch.cat([crop_box_cxcywh[:, :2], crop_box_cxcywh[:, 2:].clamp(0.04, 1.0)], dim=1)
        crop_relative_xyxy = box_cxcywh_to_xyxy(crop_box_cxcywh)
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box_xyxy.unbind(dim=-1)
        crop_w = (crop_x_max - crop_x_min).clamp_min(1e-4)
        crop_h = (crop_y_max - crop_y_min).clamp_min(1e-4)
        refined_box_xyxy = torch.stack(
            [
                crop_x_min + crop_relative_xyxy[:, 0] * crop_w,
                crop_y_min + crop_relative_xyxy[:, 1] * crop_h,
                crop_x_min + crop_relative_xyxy[:, 2] * crop_w,
                crop_y_min + crop_relative_xyxy[:, 3] * crop_h,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        box_weight = max(0.0, min(1.0, self.crop_full_logit_weight))
        box_xyxy = (box_weight * coarse_box_xyxy + (1.0 - box_weight) * refined_box_xyxy).clamp(0.0, 1.0)
        return {
            "class_logits": class_logits,
            "full_class_logits": full_logits,
            "crop_class_logits": crop_logits,
            "fusion_class_logits": fusion_logits,
            "prototype_class_logits": prototype_logits,
            "box_cxcywh": box_xyxy_to_cxcywh(box_xyxy),
            "box_xyxy": box_xyxy,
            "coarse_box_xyxy": coarse_box_xyxy,
            "coarse_box_cxcywh": coarse_box_cxcywh,
        }


class HFBirdsEfficientNetDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.25,
        model_id: str = HF_BIRDS_B2_MODEL_ID,
        multi_stage: bool = True,
        compact_dim: int = 256,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, EfficientNetForImageClassification, EfficientNetModel
        except ImportError as exc:
            raise RuntimeError(
                "The hf_birds_b2 detector requires transformers. Install T1/requirements.txt first."
            ) from exc

        self.model_id = model_id
        self.multi_stage = multi_stage
        if pretrained:
            self.features = EfficientNetForImageClassification.from_pretrained(model_id).efficientnet
            config = self.features.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.features = EfficientNetModel(config)

        channels = int(config.hidden_dim)
        stage_channels = [88, 120, 352] if multi_stage else []
        self.stage_indices = [12, 16, 23] if multi_stage else []
        self.stage_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.Hardswish(),
                )
                for in_channels in stage_channels
            ]
        )
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.compact_a = nn.Linear(channels, compact_dim, bias=False)
        self.compact_b = nn.Linear(channels, compact_dim, bias=False)
        vector_dim = channels + 4 + len(stage_channels) * 64 + compact_dim
        hidden_dim = 320
        self.classifier = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(vector_dim),
            nn.Linear(vector_dim, 160),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(160, 4),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad_(trainable)

    def _run_features(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        outputs = self.features(
            images,
            output_hidden_states=self.multi_stage,
            return_dict=True,
        )
        return outputs.last_hidden_state, outputs.hidden_states if self.multi_stage else None

    def _pool(self, images: torch.Tensor) -> torch.Tensor:
        feats, hidden_states = self._run_features(images)
        batch, channels, height, width = feats.shape
        logits = self.attention(feats).flatten(2)
        weights = torch.softmax(logits, dim=-1)
        pooled = (feats.flatten(2) * weights).sum(dim=-1)

        y_grid = torch.linspace(0.0, 1.0, height, device=images.device)
        x_grid = torch.linspace(0.0, 1.0, width, device=images.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=0).unsqueeze(0)
        center = (coords * weights).sum(dim=-1)
        spread = ((coords - center.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).sqrt()

        compact = self.compact_a(pooled) * self.compact_b(pooled)
        compact = torch.sign(compact) * torch.sqrt(compact.abs() + 1e-6)
        compact = F.normalize(compact, dim=1)

        pieces = [pooled, center, spread, compact]
        if self.multi_stage and hidden_states is not None:
            for index, projector in zip(self.stage_indices, self.stage_projectors):
                stage = projector(hidden_states[index])
                pieces.append(F.adaptive_avg_pool2d(stage, 1).flatten(1))
        return torch.cat(pieces, dim=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        vector = self._pool(images)
        class_logits = self.classifier(vector)
        box_cxcywh = torch.sigmoid(self.box_head(vector))
        box_cxcywh = torch.cat([box_cxcywh[:, :2], box_cxcywh[:, 2:].clamp(0.02, 1.0)], dim=1)
        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
        return {"class_logits": class_logits, "box_cxcywh": box_cxcywh, "box_xyxy": box_xyxy}


def build_detector(
    num_classes: int,
    pretrained: bool = True,
    backbone_name: str = "efficientnet_b0",
    detector_type: str = "attention",
    hierarchy_num_classes: int | None = None,
    crop_size: int = 260,
    crop_padding: float = 0.18,
    crop_full_logit_weight: float = 0.40,
    hf_model_id: str = HF_BIRDS_B2_MODEL_ID,
    fusion_dim: int = 704,
    prototype_dim: int = 352,
) -> nn.Module:
    if detector_type == "attention":
        return BirdDetector(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if detector_type == "bilinear_attention":
        return BilinearAttentionDetector(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if detector_type == "multiattention_part":
        return MultiAttentionPartDetector(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if detector_type == "multistage_residual":
        return MultiStageResidualDetector(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if detector_type == "crop_consistency":
        return CropConsistencyDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_name=backbone_name,
            crop_size=crop_size,
            crop_padding=crop_padding,
            crop_full_logit_weight=crop_full_logit_weight,
        )
    if detector_type in {"grid", "grid_global"}:
        return BirdGridDetector(num_classes=num_classes, pretrained=pretrained, backbone_name=backbone_name)
    if detector_type in {"shared_crop", "shared_crop_hierarchy"}:
        return SharedCropDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_name=backbone_name,
            crop_size=crop_size,
            hierarchy_num_classes=hierarchy_num_classes if detector_type == "shared_crop_hierarchy" else None,
        )
    if detector_type == "fusion_prototype_crop":
        return FusionPrototypeCropDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_name=backbone_name,
            crop_size=crop_size,
            crop_padding=crop_padding,
            crop_full_logit_weight=crop_full_logit_weight,
            fusion_dim=fusion_dim,
            prototype_dim=prototype_dim,
        )
    if detector_type == "hf_birds_b2":
        return HFBirdsEfficientNetDetector(num_classes=num_classes, pretrained=pretrained, model_id=hf_model_id)
    raise ValueError(f"Unsupported detector type: {detector_type}")


def _aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    inter_min = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    inter_max = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter = (inter_max - inter_min).clamp_min(0).prod(dim=1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=1)
    return inter / (area1 + area2 - inter + 1e-7)


def _logit_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    wrong_logits = logits.masked_fill(F.one_hot(labels, num_classes=logits.shape[1]).bool(), -torch.inf).amax(dim=1)
    return F.softplus(wrong_logits - true_logits + margin).mean()


def _arcface_loss(logits: torch.Tensor, labels: torch.Tensor, scale: float = 24.0, margin: float = 0.25) -> torch.Tensor:
    cosine = F.normalize(logits.float(), dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cosine)
    target = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    adjusted = cosine.clone()
    adjusted[target] = torch.cos(theta[target] + margin)
    return F.cross_entropy(adjusted * scale, labels, label_smoothing=0.05)


def detection_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    hierarchy_labels: torch.Tensor | None = None,
    loss_profile: str = "baseline",
) -> dict[str, torch.Tensor]:
    if "objectness_logits" in outputs:
        return grid_detection_loss(outputs, labels, boxes_xyxy)
    target_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
    class_logits = outputs["class_logits"]
    aux_cls_loss = class_logits.new_tensor(0.0)
    if "base_class_logits" in outputs and "bilinear_class_logits" in outputs:
        aux_cls_loss = 0.5 * (
            F.cross_entropy(outputs["base_class_logits"], labels, label_smoothing=0.02)
            + F.cross_entropy(outputs["bilinear_class_logits"], labels, label_smoothing=0.02)
        )
    elif "base_class_logits" in outputs and "part_class_logits" in outputs:
        aux_cls_loss = 0.5 * (
            F.cross_entropy(outputs["base_class_logits"], labels, label_smoothing=0.03)
            + F.cross_entropy(outputs["part_class_logits"], labels, label_smoothing=0.03)
        )
    elif "base_class_logits" in outputs and "stage_class_logits" in outputs:
        aux_cls_loss = 0.5 * (
            F.cross_entropy(outputs["base_class_logits"], labels, label_smoothing=0.02)
            + F.cross_entropy(outputs["stage_class_logits"], labels, label_smoothing=0.02)
        )
    elif "full_class_logits" in outputs and "crop_class_logits" in outputs:
        aux_logits = [outputs["full_class_logits"], outputs["crop_class_logits"]]
        for key in ("fusion_class_logits", "prototype_class_logits"):
            if key in outputs:
                aux_logits.append(outputs[key])
        aux_cls_loss = torch.stack(
            [F.cross_entropy(logits, labels, label_smoothing=0.08) for logits in aux_logits]
        ).mean()
    if loss_profile == "baseline":
        cls_loss = F.cross_entropy(class_logits, labels, label_smoothing=0.05)
        margin_loss = class_logits.new_tensor(0.0)
        cls_weight = 1.0
        l1_weight = 6.0
        giou_weight = 2.0
        ap_box_weight = 0.0
        coarse_l1_weight = 3.0
        coarse_giou_weight = 1.0
    elif loss_profile == "ap_focus":
        cls_loss = F.cross_entropy(class_logits, labels, label_smoothing=0.005)
        margin_loss = _logit_margin_loss(class_logits, labels, margin=0.25)
        cls_weight = 1.45
        l1_weight = 2.5
        giou_weight = 0.75
        ap_box_weight = 2.0
        coarse_l1_weight = 1.5
        coarse_giou_weight = 0.5
    elif loss_profile == "cls_focus":
        cls_loss = F.cross_entropy(class_logits, labels, label_smoothing=0.0)
        margin_loss = _logit_margin_loss(class_logits, labels, margin=0.15)
        cls_weight = 1.8
        l1_weight = 2.0
        giou_weight = 0.6
        ap_box_weight = 0.8
        coarse_l1_weight = 1.0
        coarse_giou_weight = 0.35
    elif loss_profile == "arcface":
        cls_loss = 0.55 * F.cross_entropy(class_logits, labels, label_smoothing=0.08) + 0.45 * _arcface_loss(
            class_logits,
            labels,
            scale=24.0,
            margin=0.25,
        )
        margin_loss = _logit_margin_loss(class_logits, labels, margin=0.20)
        cls_weight = 1.95
        l1_weight = 1.2
        giou_weight = 0.35
        ap_box_weight = 0.30
        coarse_l1_weight = 0.8
        coarse_giou_weight = 0.25
    else:
        raise ValueError(f"Unsupported loss profile: {loss_profile}")
    hierarchy_loss = None
    if hierarchy_labels is not None and "hierarchy_logits" in outputs:
        hierarchy_loss = F.cross_entropy(outputs["hierarchy_logits"], hierarchy_labels, label_smoothing=0.03)
    l1_loss = F.smooth_l1_loss(outputs["box_cxcywh"], target_cxcywh, beta=0.05)
    giou_loss = generalized_box_iou_loss(outputs["box_xyxy"], boxes_xyxy, reduction="mean")
    aligned_iou = _aligned_box_iou(outputs["box_xyxy"], boxes_xyxy)
    ap_box_loss = (0.55 - aligned_iou).clamp_min(0).pow(2).mean()
    coarse_l1_loss = outputs["box_xyxy"].new_tensor(0.0)
    coarse_giou_loss = outputs["box_xyxy"].new_tensor(0.0)
    if "coarse_box_cxcywh" in outputs and "coarse_box_xyxy" in outputs:
        coarse_l1_loss = F.smooth_l1_loss(outputs["coarse_box_cxcywh"], target_cxcywh, beta=0.05)
        coarse_giou_loss = generalized_box_iou_loss(outputs["coarse_box_xyxy"], boxes_xyxy, reduction="mean")
    total = (
        cls_weight * cls_loss
        + 0.35 * aux_cls_loss
        + 0.15 * margin_loss
        + l1_weight * l1_loss
        + giou_weight * giou_loss
        + ap_box_weight * ap_box_loss
        + coarse_l1_weight * coarse_l1_loss
        + coarse_giou_weight * coarse_giou_loss
    )
    result = {
        "loss": total,
        "cls_loss": cls_loss,
        "aux_cls_loss": aux_cls_loss,
        "margin_loss": margin_loss,
        "l1_loss": l1_loss,
        "giou_loss": giou_loss,
        "ap_box_loss": ap_box_loss,
        "coarse_l1_loss": coarse_l1_loss,
        "coarse_giou_loss": coarse_giou_loss,
    }
    if hierarchy_loss is not None:
        result["hierarchy_loss"] = hierarchy_loss
        result["loss"] = result["loss"] + 0.25 * hierarchy_loss
    return result


def grid_detection_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, boxes_xyxy: torch.Tensor) -> dict[str, torch.Tensor]:
    target_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
    objectness_logits = outputs["objectness_logits"]
    class_logits_map = outputs["class_logits_map"]
    box_raw_map = outputs["box_raw_map"]
    batch, height, width = objectness_logits.shape
    target_col = (target_cxcywh[:, 0] * width).floor().long().clamp(0, width - 1)
    target_row = (target_cxcywh[:, 1] * height).floor().long().clamp(0, height - 1)
    target_cell = target_row * width + target_col
    indices = torch.arange(batch, device=labels.device)

    obj_loss = F.cross_entropy(objectness_logits.flatten(1), target_cell)
    selected_class_logits = class_logits_map.permute(0, 2, 3, 1)[indices, target_row, target_col]
    local_cls_loss = F.cross_entropy(selected_class_logits, labels, label_smoothing=0.05)
    global_cls_loss = F.cross_entropy(outputs.get("global_class_logits", selected_class_logits), labels, label_smoothing=0.05)
    cls_loss = 0.35 * local_cls_loss + 0.65 * global_cls_loss

    selected_box = box_raw_map[indices, target_row, target_col]
    target_offset_x = target_cxcywh[:, 0] * width - target_col.float()
    target_offset_y = target_cxcywh[:, 1] * height - target_row.float()
    predicted_cx = (target_col.float() + torch.sigmoid(selected_box[:, 0])) / width
    predicted_cy = (target_row.float() + torch.sigmoid(selected_box[:, 1])) / height
    predicted_wh = torch.sigmoid(selected_box[:, 2:]).clamp(0.02, 1.0)
    predicted_cxcywh = torch.cat([predicted_cx[:, None], predicted_cy[:, None], predicted_wh], dim=1)
    target_offsets = torch.stack([target_offset_x, target_offset_y], dim=1)
    offset_loss = F.smooth_l1_loss(torch.sigmoid(selected_box[:, :2]), target_offsets, beta=0.05)
    size_loss = F.smooth_l1_loss(predicted_wh, target_cxcywh[:, 2:], beta=0.05)
    box_xyxy = box_cxcywh_to_xyxy(predicted_cxcywh)
    giou_loss = generalized_box_iou_loss(box_xyxy, boxes_xyxy, reduction="mean")
    total = 0.5 * obj_loss + cls_loss + 6.0 * (offset_loss + size_loss) + 2.0 * giou_loss
    return {
        "loss": total,
        "obj_loss": obj_loss,
        "cls_loss": cls_loss,
        "local_cls_loss": local_cls_loss,
        "global_cls_loss": global_cls_loss,
        "l1_loss": offset_loss + size_loss,
        "giou_loss": giou_loss,
    }


@torch.no_grad()
def evaluate_simple(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    channels_last: bool = False,
    max_batches: int | None = None,
    loss_profile: str = "baseline",
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    ious: list[float] = []
    losses: list[float] = []
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
        )
        labels = batch["label"].to(device, non_blocking=True)
        boxes = batch["box_xyxy"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(images)
            hierarchy_labels = batch.get("species_label")
            if hierarchy_labels is not None:
                hierarchy_labels = hierarchy_labels.to(device, non_blocking=True)
            loss = detection_loss(outputs, labels, boxes, hierarchy_labels=hierarchy_labels, loss_profile=loss_profile)["loss"]
        losses.append(float(loss.detach().cpu()))
        pred = outputs["class_logits"].argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        pred_boxes = outputs["box_xyxy"]
        inter_min = torch.maximum(pred_boxes[:, :2], boxes[:, :2])
        inter_max = torch.minimum(pred_boxes[:, 2:], boxes[:, 2:])
        inter = (inter_max - inter_min).clamp_min(0).prod(dim=1)
        area_pred = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp_min(0).prod(dim=1)
        area_true = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
        iou = inter / (area_pred + area_true - inter + 1e-7)
        ious.extend(iou.detach().cpu().tolist())
        if max_batches is not None and batch_index >= max_batches:
            break
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": correct / max(total, 1),
        "mean_iou": float(np.mean(ious)) if ious else math.nan,
        "ap50_proxy": float(np.mean(np.asarray(ious) >= 0.5)) if ious else math.nan,
    }


def write_prediction_csv(
    output_path: Path,
    image_ids: list[str],
    labels: list[str],
    boxes: np.ndarray,
    scores: np.ndarray,
    prediction_format: str = "object",
    coordinate_keys: str = "compact",
    prediction_encoding: str = "literal",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Id", "Predictions"]
    rows = []
    if coordinate_keys == "compact":
        key_names = ("xmin", "ymin", "xmax", "ymax")
    elif coordinate_keys == "underscore":
        key_names = ("x_min", "y_min", "x_max", "y_max")
    else:
        raise ValueError(f"Unsupported coordinate key style: {coordinate_keys}")
    for image_id, label, box, score in zip(image_ids, labels, boxes, scores):
        pred = {
            "label": label,
            "score": float(score),
            key_names[0]: float(box[0]),
            key_names[1]: float(box[1]),
            key_names[2]: float(box[2]),
            key_names[3]: float(box[3]),
        }
        value: object = [pred] if prediction_format == "list" else pred
        if prediction_encoding == "json":
            value = json.dumps(value, separators=(",", ":"))
        elif prediction_encoding != "literal":
            raise ValueError(f"Unsupported prediction encoding: {prediction_encoding}")
        rows.append({"Id": image_id, "Predictions": value})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
