from __future__ import annotations

import argparse

import torch
from torch import nn
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


def build_teacher(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name.startswith("timm:"):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install timm to use timm teacher models.") from exc
        timm_name = model_name.split(":", 1)[1]
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return efficientnet_b0(weights=weights, num_classes=num_classes)
    if model_name == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        return efficientnet_b1(weights=weights, num_classes=num_classes)
    if model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        return efficientnet_b2(weights=weights, num_classes=num_classes)
    if model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return mobilenet_v3_large(weights=weights, num_classes=num_classes)
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        return mobilenet_v3_small(weights=weights, num_classes=num_classes)

    raise ValueError(f"Unsupported teacher model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher model builder smoke helper.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    model = build_teacher(args.model_name, args.num_classes, pretrained=not args.no_pretrained)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"teacher={args.model_name} num_classes={args.num_classes} params={params}")


if __name__ == "__main__":
    main()
