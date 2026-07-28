# Condor DINOv2 Training
## Current GarmentCodeSMPLX dataset

Training now defaults to the ground-truth sample folders under:

```text
/is/cluster/fast/pachar/Data/GarmentCodeSMPLX/train/samples
```

Prepare the latest completed samples before submitting jobs:

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/prepare_smplx_data.sh
```

The preparer requires `pose1`, `pose2`, and `pose3`. It preserves raw
`design_values` and masks targets with `design_active_paths`. Because this
dataset currently has no `val/` folder, the preparer selects 10% of unique
bodies using a deterministic stratified score over garment count, garment
categories, and active categorical labels. Every garment and all three poses
belonging to a held-out body stay in validation. The selected bodies, score,
and score components are written to `split_selection.json`. Each pose is one
independent image item; the model never receives multiple poses together.
The training DataLoader uses `mode=all_images` and reshuffles those items every
epoch.

Change the holdout without changing code by setting, for example,
`VAL_FRACTION_OF_TRAIN_BODIES=0.15`. While generation is incomplete, the wrapper
writes an auditable export, but the readiness gate can still reject incomplete
or unbalanced data.

Prepared files are written to:

```text
/is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx
```

Submit any of the three models:

```bash
bash runners/condor/submit_h100.sh smplx_baseline baseline 150
bash runners/condor/submit_h100.sh smplx_multihead modelpy 150
bash runners/condor/submit_h100.sh smplx_grouped grouped 150
```

Rerun `scripts/prepare_smplx_data.sh` after more pose renders finish, then submit
a new cohort of runs so all compared models use the same frozen split.

These files run the shared `dinov2_pipeline.py` training entrypoint on one H100.


## Training While Generation Is Running

The preparer scans every sample folder and includes a garment only when its JSON,
PKL, `pose1`, `pose2`, and `pose3` files are present and readable. Unfinished
folders are listed in `balance_report.json` and skipped. The Condor runner sets
`ALLOW_UNBALANCED_DATA=1`, so this partial report is a warning and does not block
training.

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/prepare_smplx_data.sh
bash runners/condor/submit_h100.sh smplx_partial modelpy 150
```

Rerun preparation and submit a fresh run whenever you want to include newly
completed folders. Set `ALLOW_UNBALANCED_DATA=0` for final runs that should fail
unless the readiness report passes.

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

The legacy ChatGarment preparer below writes `prepared_all`. To train on it instead of the current SMPL-X default, explicitly override `PREPARED_DIR`:

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
SHARED_HIDDEN_DIM=256
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

Label smoothing and class weighting apply only to the training objective. Validation
uses plain, unweighted cross-entropy so confidence and checkpoint selection are
measured consistently.

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