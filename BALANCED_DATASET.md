# Balanced GarmentCodeSMPLX compiler

Compile every currently complete raw sample into a frozen, body-disjoint,
balanced training dataset:

```bash
cd /is/cluster/pachar/Projects/ImageToGarment
bash scripts/compile_balanced_smplx_data.sh
```

The default output is:

```text
/is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx_balanced
```

The compiler performs these steps:

1. Rejects folders unless the versioned design JSON, garment PKL, and all three
   pose PNGs are complete and readable.
2. Pools completed bodies from the raw source folders.
3. Searches for a validation-body holdout with coverage of all garment
   categories and representative categorical labels.
4. Uses the largest equal per-category quota separately in train and validation.
5. Within each category, favors samples that fill rare categorical values and
   rare 10-bin intervals of every active numeric YAML parameter.
6. Writes the standard `schema.json`, `targets.npz`, `images.json`, and
   `splits.json` consumed by training.

It does not copy images or meshes. `images.json` points to the original three
pose files, so the prepared dataset is compact.

The output also includes:

- `selection_manifest.json`: source folder, body, category, split, and design
  hash for every selected garment.
- `balance_report.json`: available and selected category counts, categorical
  label counts, deficits, rejected incomplete folders, body overlap, and the
  validation-body search audit.

To request an explicit quota instead of taking the largest balanced subset:

```bash
python compile_balanced_garmentcode_smplx.py \
  --dataset-root /is/cluster/fast/pachar/Data/GarmentCodeSMPLX \
  --out /is/cluster/fast/pachar/Data/ImageToGarment/prepared_smplx_balanced \
  --train-per-category 40 \
  --val-per-category 5 \
  --val-body-fraction 0.15 \
  --numeric-bins 10 \
  --seed 42
```

If a requested quota cannot be met, compilation stops and names the exact
category deficits. `--allow-missing-category` is available only for diagnostic
exports and should not be used for final training data.

