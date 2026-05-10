# T1 Code

This folder contains the code for our T1 submission.

Our final model is:

- an `EfficientNet-B2` detector
- plus a small crop-based classification head

## What To Download

Download the T1 dataset from the course Kaggle competition and unzip it locally.

The folder you pass to `--data-dir` should contain the official image files and split files such as:

- `classes.txt`
- `images.txt`
- `image_class_labels.txt`
- `bounding_boxes.txt`
- `train_split.txt`
- `val_split.txt`
- `test_split.txt`

## Setup

Use Python `3.10+`.

From this `code/` folder:

```bash
pip install -r requirements.txt
```

## Main Files

- `scripts/train/train_base.py`: train the detector
- `scripts/train/train_crop_head.py`: train the final crop head
- `scripts/infer/predict_final.py`: run the final model
- `scripts/infer/predict_detector.py`: run the detector only

## Included Checkpoints

This folder includes the checkpoints needed to run the final model in `checkpoints/`.

- `checkpoints/base_detector_epoch_002.pt`
  - base detector checkpoint from the final submission family
- `checkpoints/final_crop_head_best.pt`
  - final crop-head checkpoint recovered from our NCSA Delta training directory

Note:

- Use `base_detector_epoch_002.pt` together with `final_crop_head_best.pt`.

## Train The Base Detector

Example:

```bash
python scripts/train/train_base.py \
  --data-dir PATH/TO/birdwatching \
  --output-dir outputs/base_detector \
  --device cuda \
  --epochs 2 \
  --batch-size 40 \
  --eval-batch-size 96 \
  --image-size 320 \
  --detector-type attention \
  --backbone efficientnet_b2 \
  --official-validation \
  --balanced-sampler \
  --loss-profile cls_focus
```

## Train The Final Crop Head

This is the final submitted model family.

Example:

```bash
python scripts/train/train_crop_head.py \
  --data-dir PATH/TO/birdwatching \
  --checkpoint PATH/TO/base_detector_checkpoint.pt \
  --output-dir outputs/final_crop_head \
  --device cuda \
  --epochs 20 \
  --batch-size 1024 \
  --predict-batch-size 160 \
  --lr 1e-3 \
  --weight-decay 2e-4 \
  --dropout 0.15 \
  --crop-size 448 \
  --crop-padding 0.10 \
  --label-smoothing 0.02 \
  --hflip-tta \
  --score-mode margin \
  --blend-weights 0,0.15,0.25,0.35,0.50,0.65
```

## Run Final Inference

```bash
python scripts/infer/predict_final.py \
  --data-dir PATH/TO/birdwatching \
  --base-checkpoint checkpoints/base_detector_epoch_002.pt \
  --head-checkpoint checkpoints/final_crop_head_best.pt \
  --output-csv prediction.csv \
  --split test \
  --batch-size 160 \
  --device cuda \
  --hflip-tta \
  --blend-weight 0.35 \
  --score-mode margin
```

For a quick verification run, you can replace `--split test` with `--split validation`.

## Output

The output is a Kaggle submission CSV with columns:

- `Id`
- `Predictions`

Example:

```csv
Id,Predictions
000001.jpg,"{'label': '123', 'score': 0.91, 'xmin': 0.1, 'ymin': 0.2, 'xmax': 0.8, 'ymax': 0.9}"
```
