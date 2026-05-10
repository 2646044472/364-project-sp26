from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from src.detector import (
    BirdDataset,
    build_detector,
    build_label_maps,
    build_species_maps,
    count_parameters,
    data_loader_kwargs,
    detection_loss,
    evaluate_simple,
    jitter_boxes,
    load_records,
    pad_boxes,
    roi_align_single,
    set_seed,
)


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.module = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def collate_fn(batch: list[dict]) -> dict:
    output = {"image": torch.stack([item["image"] for item in batch])}
    if "label" in batch[0]:
        output["label"] = torch.stack([item["label"] for item in batch])
        output["box_xyxy"] = torch.stack([item["box_xyxy"] for item in batch])
        if "species_label" in batch[0]:
            output["species_label"] = torch.stack([item["species_label"] for item in batch])
    output["image_id"] = [item["image_id"] for item in batch]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sub-10M bird detector.")
    parser.add_argument("--data-dir", default="data/raw", help="Downloaded Kaggle competition directory.")
    parser.add_argument("--output-dir", default="outputs/mobilenetv3_small_448", help="Checkpoint directory.")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--official-validation",
        action="store_true",
        help="Use birdwatching.zip's provided val_split.txt for model selection instead of a random train split.",
    )
    parser.add_argument(
        "--provided-validation",
        action="store_true",
        help="Alias for --official-validation; uses birdwatching.zip's provided val_split.txt.",
    )
    parser.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=[
            "efficientnet_b2",
            "efficientnet_b1",
            "efficientnet_b0",
            "mobilenet_v3_large",
            "mobilenet_v3_small",
            "timm_tf_efficientnet_b2_ns",
            "timm_tf_efficientnet_b1_ns",
            "timm_tf_efficientnet_b0_ns",
            "timm_convnext_femto",
        ],
    )
    parser.add_argument(
        "--detector-type",
        default="attention",
        choices=[
            "attention",
            "bilinear_attention",
            "multiattention_part",
            "multistage_residual",
            "crop_consistency",
            "fusion_prototype_crop",
            "grid",
            "grid_global",
            "shared_crop",
            "shared_crop_hierarchy",
            "hf_birds_b2",
        ],
    )
    parser.add_argument("--crop-size", type=int, default=260, help="Internal crop size for shared_crop detectors.")
    parser.add_argument("--crop-padding", type=float, default=0.18, help="Predicted-box crop padding for crop_consistency detectors.")
    parser.add_argument("--crop-full-logit-weight", type=float, default=0.40, help="Full-image logit blend weight for crop_consistency.")
    parser.add_argument("--crop-consistency-padding-min", type=float, default=0.05)
    parser.add_argument("--crop-consistency-padding-max", type=float, default=0.30)
    parser.add_argument("--crop-consistency-center-jitter", type=float, default=0.12)
    parser.add_argument(
        "--hf-model-id",
        default="dennisjooo/Birds-Classifier-EfficientNetB2",
        help="Hugging Face model id for the bird-specific EfficientNet-B2 backbone.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze detector backbones with set_backbone_trainable() for the first N epochs.",
    )
    parser.add_argument(
        "--teacher-crop-epochs",
        type=int,
        default=0,
        help="Use ground-truth boxes for the crop branch during the first N epochs of shared_crop training.",
    )
    parser.add_argument("--seed", "--random-seed", dest="seed", type=int, default=364)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None, help="Load model weights from an existing checkpoint before training.")
    parser.add_argument("--partial-resume", action="store_true", help="Load only matching parameter names/shapes from --resume-checkpoint.")
    parser.add_argument("--train-on-validation", action="store_true", help="Include birdwatching.zip's val_split.txt labels in final training.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory on Windows.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Sample rare classes more often during training.")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="Use EMA weights for validation/checkpointing when > 0.")
    parser.add_argument(
        "--loss-profile",
        default="baseline",
        choices=["baseline", "ap_focus", "cls_focus", "arcface"],
        help="Use baseline training loss or AP-oriented classification/bbox weighting.",
    )
    parser.add_argument(
        "--bbox-crop-prob",
        type=float,
        default=0.0,
        help="Probability of training on a padded ground-truth bird crop before resize.",
    )
    parser.add_argument(
        "--aux-crop-cls-weight",
        type=float,
        default=0.0,
        help="Extra classification loss weight on GPU crops from ground-truth boxes; keeps bbox loss on full images.",
    )
    parser.add_argument("--aux-crop-padding", type=float, default=0.12, help="Padding for --aux-crop-cls-weight crops.")
    parser.add_argument("--aux-crop-chunk-size", type=int, default=32, help="Chunk size for aux crop classification to avoid VRAM spikes.")
    parser.add_argument("--deterministic", action="store_true", help="Prefer deterministic kernels; this can reduce GPU throughput.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug/quick-tune limit for training batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Quick-tune limit for validation batches.")
    parser.add_argument(
        "--max-gpu-memory-fraction",
        type=float,
        default=0.70,
        help="Cap this process below dedicated VRAM so Windows does not spill into Shared GPU Memory.",
    )
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels_last tensors on CUDA.")
    parser.add_argument("--save-last", action="store_true", help="Save last.pt after every epoch in addition to best.pt.")
    parser.add_argument("--save-epoch-checkpoints", action="store_true", help="Save epoch_XXX.pt after every epoch for AP-based model selection.")
    parser.add_argument("--submission-name", default=None, help="Optional basename for a submission generated from this run.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.official_validation = args.official_validation or args.provided_validation

    set_seed(args.seed, deterministic=args.deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        if args.max_gpu_memory_fraction <= 0 or args.max_gpu_memory_fraction > 1:
            raise ValueError("--max-gpu-memory-fraction must be in the range (0, 1].")
        if args.max_gpu_memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_fraction, cuda_index)

    root = Path(args.data_dir)
    records = load_records(root, "train")
    official_val_records = None
    if args.official_validation and not args.train_on_validation:
        official_val_records = load_records(root, "validation")
    if args.train_on_validation:
        try:
            val_records = load_records(root, "validation")
            seen = {record.image_path for record in records}
            records.extend(record for record in val_records if record.image_path not in seen)
        except Exception as exc:
            print(f"Validation merge skipped: {exc}")
    if args.smoke_test:
        records = records[: min(len(records), 128)]
        if official_val_records is not None:
            official_val_records = official_val_records[: min(len(official_val_records), 128)]

    label_to_idx, idx_to_label = build_label_maps(records)
    species_to_idx, idx_to_species, class_to_species = build_species_maps(idx_to_label)
    if official_val_records is None:
        val_size = max(1, int(len(records) * args.val_fraction))
        if len(records) <= val_size:
            raise RuntimeError("Not enough records to create a training and validation split.")
        train_size = len(records) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(records), generator=generator).tolist()
        val_indices = set(permutation[:val_size])
        train_records = [record for index, record in enumerate(records) if index not in val_indices]
        val_records = [record for index, record in enumerate(records) if index in val_indices]
        validation_source = "random"
    else:
        train_records = records
        val_records = [record for record in official_val_records if record.label in label_to_idx]
        train_size = len(train_records)
        val_size = len(val_records)
        validation_source = "official"
        if not val_records:
            raise RuntimeError("Official validation split did not contain labels present in training.")
    class_to_species_arg = class_to_species if args.detector_type == "shared_crop_hierarchy" else None
    train_dataset = BirdDataset(
        train_records,
        label_to_idx,
        args.image_size,
        training=True,
        class_to_species=class_to_species_arg,
        bbox_crop_prob=args.bbox_crop_prob,
    )
    val_dataset = BirdDataset(val_records, label_to_idx, args.image_size, training=False, class_to_species=class_to_species_arg)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    train_shuffle = True
    if args.balanced_sampler:
        class_counts = Counter(record.label for record in train_records)
        weights = torch.as_tensor([1.0 / class_counts[record.label] for record in train_records], dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=train_generator)
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **data_loader_kwargs(args.num_workers, args.prefetch_factor, device, train_generator, pin_memory=not args.no_pin_memory),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **data_loader_kwargs(args.num_workers, args.prefetch_factor, device, pin_memory=not args.no_pin_memory),
    )

    model = build_detector(
        num_classes=len(idx_to_label),
        pretrained=not args.no_pretrained,
        backbone_name=args.backbone,
        detector_type=args.detector_type,
        hierarchy_num_classes=len(idx_to_species) if args.detector_type == "shared_crop_hierarchy" else None,
        crop_size=args.crop_size,
        crop_padding=args.crop_padding,
        crop_full_logit_weight=args.crop_full_logit_weight,
        hf_model_id=args.hf_model_id,
    ).to(device)
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_labels = checkpoint.get("idx_to_label")
        if checkpoint_labels != idx_to_label:
            raise RuntimeError("Resume checkpoint labels do not match the current training label map.")
        if args.partial_resume:
            model_state = model.state_dict()
            source_state = checkpoint["model"]
            matched = {
                key: value
                for key, value in source_state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            model_state.update(matched)
            model.load_state_dict(model_state)
            print(f"Partially resumed {len(matched)} tensors from {args.resume_checkpoint}")
        else:
            model.load_state_dict(checkpoint["model"])
            print(f"Resumed model weights from {args.resume_checkpoint}")
    use_channels_last = device.type == "cuda" and not args.no_channels_last
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    params = count_parameters(model)
    if params > 10_000_000:
        raise RuntimeError(f"Model has {params:,} parameters, exceeding the 10M limit.")
    print(f"Train records: {train_size}, validation records: {val_size} ({validation_source}), classes: {len(idx_to_label)}")
    print(
        f"Seed: {args.seed}, workers: {args.num_workers}, prefetch_factor: {args.prefetch_factor}, "
        f"channels_last: {use_channels_last}, balanced_sampler: {args.balanced_sampler}, "
        f"ema_decay: {args.ema_decay}, pin_memory: {not args.no_pin_memory}, "
        f"max_gpu_memory_fraction: {args.max_gpu_memory_fraction}, device: {device}"
    )
    if device.type == "cuda":
        total_mib = torch.cuda.get_device_properties(device).total_memory / 1024**2
        capped_mib = total_mib * args.max_gpu_memory_fraction
        print(f"Dedicated VRAM cap for this process: {capped_mib:.0f} MiB / {total_mib:.0f} MiB")
    print(f"Model parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=steps,
        pct_start=0.1,
        final_div_factor=20,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    ema = ModelEma(model, args.ema_decay) if args.ema_decay > 0 else None

    best_score = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        if hasattr(model, "set_backbone_trainable"):
            backbone_trainable = epoch > args.freeze_backbone_epochs
            model.set_backbone_trainable(backbone_trainable)
            if epoch == 1 or epoch == args.freeze_backbone_epochs + 1:
                print(f"Backbone trainable: {backbone_trainable}")
        model.train()
        running = []
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1):
            images = batch["image"].to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last if use_channels_last else torch.contiguous_format,
            )
            labels = batch["label"].to(device, non_blocking=True)
            boxes = batch["box_xyxy"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                if args.detector_type in {"shared_crop", "shared_crop_hierarchy"} and epoch <= args.teacher_crop_epochs:
                    outputs = model(images, crop_boxes=boxes)
                elif args.detector_type == "crop_consistency":
                    crop_boxes = jitter_boxes(
                        boxes,
                        padding_min=args.crop_consistency_padding_min,
                        padding_max=args.crop_consistency_padding_max,
                        center_jitter=args.crop_consistency_center_jitter,
                    )
                    outputs = model(images, crop_boxes=crop_boxes)
                else:
                    outputs = model(images)
                hierarchy_labels = batch.get("species_label")
                if hierarchy_labels is not None:
                    hierarchy_labels = hierarchy_labels.to(device, non_blocking=True)
                losses = detection_loss(outputs, labels, boxes, hierarchy_labels=hierarchy_labels, loss_profile=args.loss_profile)
                if args.aux_crop_cls_weight > 0:
                    crop_boxes = pad_boxes(boxes, padding=args.aux_crop_padding)
                    crops = roi_align_single(images, crop_boxes, args.image_size)
                    crop_losses = []
                    chunk_size = max(1, args.aux_crop_chunk_size)
                    for start in range(0, crops.shape[0], chunk_size):
                        stop = min(start + chunk_size, crops.shape[0])
                        crop_outputs = model(crops[start:stop])
                        crop_losses.append(
                            torch.nn.functional.cross_entropy(
                                crop_outputs["class_logits"],
                                labels[start:stop],
                                label_smoothing=0.02,
                            )
                            * (stop - start)
                        )
                    crop_cls_loss = torch.stack(crop_losses).sum() / crops.shape[0]
                    losses["crop_cls_loss"] = crop_cls_loss
                    losses["loss"] = losses["loss"] + args.aux_crop_cls_weight * crop_cls_loss
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            scheduler.step()
            running.append(float(losses["loss"].detach().cpu()))
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

        eval_model = ema.module if ema is not None else model
        should_eval = epoch == args.epochs or epoch % max(args.eval_every, 1) == 0
        metrics = (
            evaluate_simple(
                eval_model,
                val_loader,
                device,
                channels_last=use_channels_last,
                max_batches=args.max_val_batches,
                loss_profile=args.loss_profile,
            )
            if should_eval
            else {"loss": float("nan"), "accuracy": 0.0, "mean_iou": 0.0, "ap50_proxy": 0.0}
        )
        score = metrics["accuracy"] * 0.35 + metrics["mean_iou"] * 0.65
        memory_metrics = {}
        if device.type == "cuda":
            memory_metrics = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
            torch.cuda.reset_peak_memory_stats(device)
        row = {
            "epoch": epoch,
            "train_loss": sum(running) / max(len(running), 1),
            **metrics,
            **memory_metrics,
            "selection_score": score,
            "checkpoint_weights": "ema" if ema is not None else "raw",
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        checkpoint = {
            "model": eval_model.state_dict(),
            "raw_model": model.state_dict() if ema is not None else None,
            "idx_to_label": idx_to_label,
            "label_to_idx": label_to_idx,
            "idx_to_species": idx_to_species,
            "species_to_idx": species_to_idx,
            "class_to_species": class_to_species,
            "args": vars(args),
            "params": params,
            "metrics": row,
        }
        if args.save_last:
            torch.save(checkpoint, output_dir / "last.pt")
        if args.save_epoch_checkpoints:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if should_eval and score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_score": best_score, "params": params}, handle, indent=2)


if __name__ == "__main__":
    main()
