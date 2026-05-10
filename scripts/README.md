# Scripts

This folder contains the command-line entry points.

If you want to run the code, start here.

## Training

- `train/train_base.py`
  - trains the base `EfficientNet-B2` detector
- `train/train_crop_head.py`
  - trains the final crop-based classification head on top of a frozen detector

## Inference

- `infer/predict_detector.py`
  - runs the detector only
- `infer/predict_final.py`
  - runs the final submission model

## Recommended Order

1. Train the base detector with `train/train_base.py`
2. Train the crop head with `train/train_crop_head.py`
3. Generate the final submission with `infer/predict_final.py`

These scripts are thin wrappers around the real implementation in `../src/`.

You can also use the example checkpoints included in `../checkpoints/` to run inference directly.
