# Train on H100 and compare against GarmentImage

## 1. Compile the balanced dataset

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/compile_balanced_smplx_data.sh
```

The H100 payload refuses to train unless
`prepared_smplx_balanced/balance_report.json` has `ready: true`.

## 2. Submit an H100 training job

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash runners/condor/submit_garment_tree_balanced_h100.sh garment_tree_v1 150
```

The second argument is the cluster bid. The command prints the exact log and
run directories. The best checkpoint is `<printed run directory>/best.pt`.

The job requests one H100 with at least 80 GB device memory, 8 CPU cores, 64 GB
system memory, and 20 GB scratch disk. It trains only from the frozen balanced
prepared dataset and does not override a failed balance audit.

## 3. Smoke-test GarmentImage inference

Run this on a GPU node after training:

```bash
python scripts/garmentimages_batch_infer_garment_tree.py \
  --checkpoint runs/<training-run>/best.pt \
  --out-dir runs/garmentimage_comparisons/<training-run> \
  --device cuda \
  --render-python venv/bin/python \
  --garmentcode-dir GarmentCodeRC \
  --limit 5
```

The image root defaults to:

```text
/is/cluster/fast/pachar/Data/GarmentImage
```

Use `--image-root` to override it.

## 4. Run the full comparison

```bash
python scripts/garmentimages_batch_infer_garment_tree.py \
  --checkpoint runs/<training-run>/best.pt \
  --out-dir runs/garmentimage_comparisons/<training-run> \
  --device cuda \
  --render-python venv/bin/python \
  --garmentcode-dir GarmentCodeRC \
  --skip-existing
```

The adapter reuses the existing batch pipeline. Results are written to:

```text
<out-dir>/yaml/
<out-dir>/front_preview/
<out-dir>/side_by_side/
<out-dir>/summary.csv
```

`summary.csv` records success or the rendering error for every input image.

