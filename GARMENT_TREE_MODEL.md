# GarmentTreeNet

GarmentTreeNet reconstructs a complete GarmentCode design from one RGB image.
It is trained end to end on the prepared GarmentCodeSMPLX data and does not need
multiple images at inference.

## Data contract

Prepare the live dataset with the existing body-disjoint, balance-audited
converter:

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/prepare_smplx_data.sh
```

The current contract contains 67 normalized numeric targets and 22 categorical
targets. Inactive topology branches are masked. The split is by body, never by
image or garment, and the three pose renders of a garment cannot cross splits.

## Model

The network has four stages:

1. A trainable multi-scale image encoder captures small contour details and the
   full garment silhouette.
2. Twelve learned part queries separately read global topology, waistband,
   shirt, collar, sleeves, asymmetry, each skirt family, and pants.
3. Categorical topology is predicted first.
4. Numeric branches are conditioned on the soft topology distribution and emit
   a bounded value, uncertainty, and activity probability for every parameter.

Training is still single-image reconstruction. When enabled, a second pose of
the same garment is used only for prediction-consistency regularization; each
individual forward call receives one image. Validation evaluates every pose as
an independent one-image prediction.

## Train

Submit the default H100 run:

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash runners/condor/submit_garment_tree_h100.sh garment_tree_v1 150
```

Or run directly:

```bash
python train_garment_tree.py \
  --prepared-dir /is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx \
  --out-dir runs/garment_tree_v1 \
  --device cuda --amp \
  --allow-unbalanced-data
```

`--allow-unbalanced-data` is appropriate only while render generation is still
in progress. Remove it for the final dataset so a failed balance audit stops the
run. The best exponential-average checkpoint is `best.pt`; `history.jsonl`
contains all train/validation curves.

## Infer from exactly one image

```bash
python infer_garment_tree.py \
  --checkpoint runs/garment_tree_v1/best.pt \
  --image /path/to/one_garment_image.png \
  --yaml-out output/design.yaml \
  --json-out output/design_confidence.json \
  --device cuda
```

The YAML is directly consumable by GarmentCode. The companion JSON records
topology confidence, per-parameter activity probability, and numeric uncertainty
so weak parameter families can be identified for targeted data generation.

