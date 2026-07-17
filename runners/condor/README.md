# Condor DINOv2 Training

These files run the shared `dinov2_pipeline.py` training entrypoint on one H100.

## 1. Fetch and prepare all data

Run this once on the cluster:

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/fetch_all_fast_data.sh
```

By default this downloads the supported ChatGarment dataset versions from the Hugging Face dataset into:

```text
/is/cluster/fast/pachar/Data/ChatGarmentDataset
```

It includes:

```text
v1: training/synthetic/data_restpose_img_v1.json + garments_imgs_v1_1.zip ... garments_imgs_v1_5.zip
v2: training/synthetic/data_img_v2.json          + garments_imgs_v2_1.zip ... garments_imgs_v2_3.zip
v3: training/synthetic/data_img_v3.json          + garments_imgs_v3.zip
v4: training/synthetic/data_img_v4.json          + garments_imgs_v4.zip
```

It writes the compact prepared dataset used by training to:

```text
/is/cluster/fast/pachar/Data/ImageToGarment/prepared_all
```

The v1 rest-pose shards are very large. To skip v1 and prepare v2/v3/v4 only:

```bash
INCLUDE_V1=0 bash scripts/fetch_all_fast_data.sh
```

To prepare only a subset, set the version toggles:

```bash
INCLUDE_V1=0 INCLUDE_V2=1 INCLUDE_V3=0 INCLUDE_V4=0 \
PREPARED_DIR=/is/cluster/fast/pachar/Data/ImageToGarment/prepared_v2 \
bash scripts/fetch_all_fast_data.sh
```

To keep every pose frame instead of only frame `0`:

```bash
FRAMES=all bash scripts/fetch_all_fast_data.sh
```

The older v2-only helper still exists for quick checks:

```bash
bash scripts/fetch_fast_data.sh
```

## 2. Submit training

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
mkdir -p runners/condor/logs
bash runners/condor/submit_h100.sh garment_multihead_all modelpy
```

Training uses `/is/cluster/fast/pachar/Data/ImageToGarment/prepared_all` by default. To force a different prepared dataset:

```bash
PREPARED_DIR=/is/cluster/fast/pachar/Data/ImageToGarment/prepared_v2 \
bash runners/condor/submit_h100.sh garment_multihead_v2 modelpy
```

The submit wrapper uses HOOD-style bidding with `condor_submit_bid 100`. To run the baseline:

```bash
bash runners/condor/submit_h100.sh garment_baseline_all baseline
```

To change the bid, pass it as the third argument:

```bash
bash runners/condor/submit_h100.sh garment_multihead_all modelpy 150
```

The requested resources are:

```text
1 H100 GPU
8 CPU cores
64 GB CPU RAM
20 GB disk
```

The default Condor run trains for 1000 epochs and writes `epoch_0100.pt`, `epoch_0200.pt`, etc. in addition to `best.pt` and `last.pt`.

The H100 runner now uses regularized defaults for the overfitting regime:

```text
DROPOUT=0.4
HEAD_HIDDEN_DIMS=64 32
WEIGHT_DECAY=3e-3
HEAD_LAYER_NORM=1
AUGMENTATION=light
LABEL_SMOOTHING=0.05
CLASS_WEIGHTING=effective
CLASS_WEIGHT_BETA=0.999
CLASS_WEIGHT_MAX=5.0
REG_LOSS=smooth_l1
GRAD_CLIP_NORM=1.0
LR_PLATEAU_PATIENCE=5
EARLY_STOPPING_PATIENCE=20
```

Override any of these in the submit environment or before calling the payload if validation loss starts underfitting.

Outputs are written under:

```text
/is/cluster/pachar/Projects/ImageToGarment/runs
```

Condor logs are written under:

```text
runners/condor/logs
```

## W&B Loss Plots

Condor training enables Weights & Biases by default:

```text
WANDB=1
WANDB_PROJECT=ImageToGarment
WANDB_MODE=online
```

The training script logs epoch curves for:

```text
train/loss, train/loss_reg, train/loss_cat, train/cat_acc
val/loss, val/loss_reg, val/loss_cat, val/cat_acc
gap/loss, gap/loss_reg, gap/loss_cat
best/val_loss
optim/lr
```

If the cluster is not already logged into W&B, create a local ignored file:

```bash
cat > runners/condor/secrets.sh <<'EOF'
export WANDB_API_KEY=your_key_here
EOF
```

To disable W&B for a run, edit the submit file environment or run the payload with:

```bash
WANDB=0 bash runners/condor/run_train_dinov2_h100.sh --architecture modelpy
```