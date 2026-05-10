# Source Code

This folder contains the main implementation.

If you want to understand how the model works, start here.

## Suggested Reading Order

1. `detector.py`
2. `train.py`
3. `train_crop_feature_head.py`
4. `predict_final.py`

## File Guide

- `detector.py`
  - the largest file in this folder
  - contains dataset loading, model building, utility functions, prediction helpers, and CSV writing
  - this is the best place to start if you want to understand the full pipeline

- `train.py`
  - training code for the base bird detector
  - uses the detector model defined in `detector.py`

- `train_crop_feature_head.py`
  - training code for the final crop-feature head
  - this is the main file for the final submitted model

- `predict_final.py`
  - inference code for the final model
  - loads both the detector checkpoint and the crop-head checkpoint

- `predict_checkpoint.py`
  - inference code for detector-only predictions

- `train_teacher_crop_classifier.py`
  - helper code for loading a larger teacher model
  - this is only used for optional training-time distillation

- `calculate_AP.py`
  - local evaluation helper for AP

## Final Submitted Model

The final submitted system uses:

- the detector from `train.py`
- the crop head from `train_crop_feature_head.py`
- the inference pipeline from `predict_final.py`
