# ImageToGarment

Reconstruct a complete **GarmentCode sewing-pattern design** from a **single RGB image**.

The repository covers the full path from raw renders to a draped 3D garment:

```
raw dataset ──► prepared data ──► model ──► design.yaml ──► GarmentCode drape + render
 (ChatGarment /  (schema.json,    (DINOv2 heads /  (prediction_to_yaml)   (render_garmentcode)
  GarmentCodeSMPLX) targets.npz)    tree / ensemble)
```

Every model predicts the same target contract: **normalized numeric parameters** (with an
active/inactive mask) plus **categorical topology selectors** (`meta.upper`, `meta.wb`,
`meta.bottom`, cuff types, …). Because the schema is a *union* over all garment families,
any single garment only activates the slots reachable from its own topology.

---

## Contents

| Area | Entry points |
| --- | --- |
| Data preparation | `prepare_data.py`, `prepare_garmentcode_smplx.py`, `compile_balanced_garmentcode_smplx.py`, `compile_output_balanced_garmentcode_smplx.py`, `compile_chatgarment_auxiliary.py` |
| DINOv2 regression models | `dinov2_pipeline.py` (`baseline` / `modelpy` / `grouped`) |
| Tree-structured model | `train_garment_tree.py`, `garment_tree_model.py` |
| Route-constrained ensemble | `train_garment_ensemble.py`, `garment_ensemble_model.py` |
| Text-instruction editing (LLM) | `prepare_edit_data.py`, `train_edit_model.py`, `edit_model.py` |
| Mesh fitting / TRELLIS.2 route | `mesh_fitting_gnn.py`, `scripts/person_garment_trellis_gnn_pipeline.py` |
| Decoding + rendering | `prediction_to_yaml.py`, `render_garmentcode.py` |
| Cluster jobs | `runners/condor/`, `runners/slurm/batch/` |

Companion documents: [`GARMENT_TREE_MODEL.md`](GARMENT_TREE_MODEL.md),
[`GARMENT_TREE_RUN.md`](GARMENT_TREE_RUN.md), [`BALANCED_DATASET.md`](BALANCED_DATASET.md),
[`runners/condor/README.md`](runners/condor/README.md),
[`docs/TRELLIS2_CLUSTER_SETUP.md`](docs/TRELLIS2_CLUSTER_SETUP.md).

---

## Requirements

Python 3.11 with `torch`, `torchvision`, `numpy`, `Pillow`, `pyyaml`.
Additionally: `wandb` (optional logging), `matplotlib` (`scripts/plot_history.py`),
`transformers` + `peft` + `bitsandbytes` (editing model), `pygarment` / a local
`GarmentCodeRC` checkout (rendering), `pytest` (tests).

Neither GarmentCode nor DINOv2 nor the datasets are vendored — they are expected as local
checkouts/downloads (see `.gitignore`).

---

## 1. Data

Two ground-truth sources are supported, both reduced to the same four files.

### 1a. ChatGarment (`prepare_data.py`)

Turns the [ChatGarment dataset](https://huggingface.co/datasets/sy000/ChatGarmentDataset)
(`sy000/ChatGarmentDataset`) into a frozen train/val/test split.

```bash
python prepare_data.py \
  --json       data/chatgarment_data/training/synthetic/data_img_v2.json \
  --image-root data/chatgarment_data/garments_imgs_v2_3 \
  --tag        v2 \
  --out        prepared_v2
```

Multiple versions must be paired explicitly, because **gid numbers are reused across
versions** (`gid 1` in v2 ≠ `gid 1` in v3):

```bash
python prepare_data.py \
  --set v2 .../data_img_v2.json .../garments_imgs_v2_1 .../garments_imgs_v2_3 \
  --set v3 .../data_img_v3.json .../garments_imgs_v3 \
  --out prepared_all
```

Each `--set TAG JSON ROOT…` is scanned in isolation; the schema is still the union over all
sets, so slot indices stay stable. Use `--set` **or** `--json/--image-root`, never both.

Key flags: `--frames` (default `0`; `all` for every pose — images only, never the schema),
`--val`/`--test` (0.1/0.1), `--seed` (42), `--limit N` (smoke test).

On the cluster, `bash scripts/fetch_all_fast_data.sh` fetches v1–v4 and builds the combined
manifest (`INCLUDE_V1=0` skips the very large rest-pose shards).

### 1b. GarmentCodeSMPLX (`prepare_garmentcode_smplx.py`)

The current default source: rendered ground-truth samples with `pose1`/`pose2`/`pose3`, a
versioned design JSON, and a garment PKL. A folder is accepted only when all of those are
present and readable.

```bash
bash scripts/prepare_smplx_data.sh          # → prepared_smplx
```

Splits are **by body**, never by image or garment — all three poses of every garment
belonging to a held-out body stay in validation. When no source `val/` exists, 10 % of
bodies are selected by a deterministic stratified score (garment count, categories, active
categorical labels), audited in `split_selection.json`. Each pose is one independent
single-image item; the model never receives multiple poses at once.

### 1c. Balanced compilers

| Compiler | Selection strategy |
| --- | --- |
| `compile_balanced_garmentcode_smplx.py` | Equal per-category quota in train and validation; within a category, favours rare categorical values and rare numeric bins. Fails loudly on category deficits. See [`BALANCED_DATASET.md`](BALANCED_DATASET.md). |
| `compile_output_balanced_garmentcode_smplx.py` | Keeps **every** complete garment and balances exposure in *model-output space* — active heads, categorical classes, binned numeric values. |
| `compile_chatgarment_auxiliary.py` | Maps ChatGarment prepared data onto an existing GarmentCode schema for auxiliary pretraining. |

Both compilers emit `selection_manifest.json` and `balance_report.json`; training refuses to
start unless `balance_report.json` has `ready: true` (override with `--allow-unbalanced-data`
only while render generation is still running).

### Outputs (identical for every preparer)

| File | Contents |
| --- | --- |
| `schema.json` | `cont_slots`, `const_slots`, `const_ranges`, `cat_vocab` — the meaning of every index |
| `targets.npz` | `gids`, `y_cont` + `mask`, `y_const` + `const_mask`, `y_cat` (`-1` = inactive) — one row per garment |
| `images.json` | per-garment pose/view image paths (files are referenced, never copied) |
| `splits.json` | frozen garment/body-level split with seed and ratios |

Use `GarmentDataset` from `prepare_data.py` to consume them:

```python
from prepare_data import GarmentDataset
ds = GarmentDataset("prepared_smplx", split="train", mode="all_images", train=True)
# batch: image [B,3,H,W], y_cont, mask, y_const, const_mask, y_cat, gid
```

`mode="single"` samples one view per garment per epoch; `mode="all_images"` treats every
image as its own item while keeping all views of a garment in the same split.

---

## 2. Models

### 2a. DINOv2 regression heads — `dinov2_pipeline.py`

One entry point, three architectures sharing data, checkpoints, YAML export and metrics:

| `--architecture` | Head design | Implementation |
| --- | --- | --- |
| `baseline` | two flat heads: one regression head over `y_cont`+`y_const`, one categorical head | `train_dinov2.py` |
| `modelpy` | one small MLP **per parameter**, typed from the GarmentCode schema | `model.py` |
| `grouped` | shared trunk → three semantic MLPs (upper / lower / waistband) | `train_dinov2_grouped.py` |

```bash
python dinov2_pipeline.py train \
  --architecture grouped \
  --prepared-dir prepared_smplx \
  --out-dir runs/garment_grouped \
  --backbone dinov2_vitl14 \
  --mode all_images --epochs 150 --batch-size 32 --device cuda --amp
```

Defaults worth knowing: frozen DINOv2-L backbone, light augmentation, dropout 0.1,
SmoothL1 regression, effective-number class weighting, label smoothing 0.05, gradient
clipping, `ReduceLROnPlateau`, early stopping (patience 30), checkpoints every 100 epochs.
`--unfreeze-backbone` fine-tunes the encoder; `--wandb` logs curves.
Grouped widths: `--hidden-dim` (trunk, 512), `--branch-hidden-dim` (128),
`--waistband-hidden-dim` (64).

Inference auto-detects the architecture stored in the checkpoint:

```bash
python dinov2_pipeline.py infer \
  --checkpoint runs/garment_grouped/best.pt \
  --image path/to/image.png \
  --yaml-out output/design.yaml \
  --render-3d --device cuda
```

`--render-upper/--render-wb/--render-bottom` override the inferred topology for rendering,
which is useful for isolating topology errors from parameter errors.

### 2b. GarmentTreeNet — `train_garment_tree.py`

A network that mirrors the GarmentCode parameter tree:

1. trainable multi-scale image encoder (contour detail + full silhouette);
2. twelve learned **part queries** (global, waistband, shirt, collar, sleeve, asymmetry, each
   skirt family, pants);
3. categorical **topology decoded first**;
4. numeric branches conditioned on the soft topology distribution, each emitting a bounded
   value, an uncertainty, and an activity probability.

```bash
python train_garment_tree.py \
  --prepared-dir prepared_smplx_balanced \
  --out-dir runs/garment_tree_v1 \
  --device cuda --amp
```

Training stays single-image; a second pose is used only as a prediction-consistency
regularizer, and validation scores every pose independently. `best.pt` is the best
exponential-moving-average checkpoint; `history.jsonl` holds all curves.
`train_garment_tree_output_balanced.py` is the same loop driven by the output-balanced
sampler.

### 2c. Route-constrained ensemble — `train_garment_ensemble.py`

Five bagged members over a **joint route distribution** across the three root selectors
(`meta.upper`, `meta.wb`, `meta.bottom`). Only legal routes exist in the table, and root
logits are recovered by marginalising the joint distribution — so a member can never emit an
impossible topology combination. On top of that: a shared frozen patch encoder with a cached
feature bank, out-of-fold training for the stacker, router warm-up then freezing, pose
consistency, EMA weights, and optional auxiliary pretraining on ChatGarment-derived data
(`--aux-prepared-dir`).

```bash
python train_garment_ensemble.py \
  --prepared-dir prepared_smplx_balanced \
  --out-dir runs/garment_tree_ensemble \
  --members 5 --folds 5 --epochs 180 --device cuda
```

### 2d. Text-instruction editing — `train_edit_model.py`

Instruction-conditioned garment editing with a QLoRA decoder LLM that **regresses** floats
from hidden states instead of emitting digits. Two readout variants share the backbone, data
and losses, so they are directly comparable:

* `--variant single_token` (A, ChatGarment's scheme): one `<ALLNUM>` sentinel closes the
  numeric section; its hidden state feeds an MLP that emits the whole `N_SLOTS` vector.
* `--variant per_token` (B): each active float renders as one `<VAL>` token preceded by its
  key; each `<VAL>` hidden state feeds a **shared** MLP emitting one scalar.

```bash
python prepare_edit_data.py --out prepared_edit          # 76-slot canonical layout
python train_edit_model.py --variant per_token --out runs/edit_per_token
python infer_edit.py --ckpt runs/edit_per_token/final --out preds.jsonl
python eval_edit.py --preds preds.jsonl
python edit_to_design.py --preds preds.jsonl --template <design.yaml>
```

The 76 editing slots are *derived* from the frozen 152-slot image schema by stripping the
body prefix — never hand-written. Design invariants: regression heads are plain linear (no
sigmoid/tanh — clamping happens at inference only, so genuine 0.0/1.0 boundary values remain
reachable), the new special-token embedding rows are trainable via peft's
`trainable_token_indices`, and hidden states are indexed **at** the special token's own
position. Loss is `CE(target JSON tokens) + λ · L1(regressed floats)`.

### 2e. Mesh route (TRELLIS.2 + GNN)

`scripts/person_garment_trellis_gnn_pipeline.py` chains: person image → garment-only image
(SegFormer clothes segmentation) → TRELLIS.2 mesh → GarmentCode mesh → MeshGraphNets-style
template deformation (`mesh_fitting_gnn.py`, topology-agnostic losses). Each stage writes
files so it can be inspected or rerun independently. TRELLIS.2 is external — see
[`docs/TRELLIS2_CLUSTER_SETUP.md`](docs/TRELLIS2_CLUSTER_SETUP.md).

### 2f. ChatGarment VLM comparison baseline

`scripts/prepare_chatgarment_smplx_vlm.py` exports GarmentCodeSMPLX as ChatGarment-format
manifests with normalized sparse `[SEG]` targets (train/val bodies disjoint), for LLaVA-7B
LoRA training via `runners/condor/submit_chatgarment_smplx_h100.sh`. This exists to test
whether a large VLM is actually necessary for the task.

---

## 3. Decoding predictions back to GarmentCode

```
continuous ([SEG]):  raw = lo + pred * (hi - lo)     # lo,hi from GarmentCode default.yaml
constants:           raw = lo + pred * (hi - lo)     # lo,hi from schema.json const_ranges
categoricals:        value = cat_vocab[field][argmax(logits_field)]
```

`prediction_to_yaml.py` (or the integrated `dinov2_pipeline.py infer` path) writes the design
YAML, casting by the **declared type** from the GarmentCode template.

> **Do not blindly round all constants.** Constants sit on integer grids *except* the
> float-valued `flare-skirt.skirt-many-panels.panel_curve` (range −0.35…0.45) under both
> `lowerbody_garment` and `wholebody_garment`. Round only `type: int` params. `schema.json`
> does not yet record int/float types — a standalone decoder must look them up in
> `assets/design_params/default.yaml`.

Render the result headlessly:

```bash
python render_garmentcode.py --design output/design.yaml --out-dir runs/garmentcode_reconstructions
```

---

## 4. Evaluating on GarmentImage

Batch inference + side-by-side renders against the GarmentImage dataset:

```bash
# DINOv2 architectures
python scripts/garmentimages_batch_infer.py \
  --checkpoint runs/<run>/best.pt --architecture grouped \
  --image-root /path/to/GarmentImage \
  --out-dir runs/garmentimage_comparisons/grouped \
  --device cuda --render-python venv/bin/python --garmentcode-dir GarmentCodeRC

# tree / ensemble checkpoints
python scripts/garmentimages_batch_infer_garment_tree.py \
  --checkpoint runs/<run>/best.pt \
  --out-dir runs/garmentimage_comparisons/<run> \
  --device cuda --render-python venv/bin/python --garmentcode-dir GarmentCodeRC --skip-existing
```

Outputs land in `<out-dir>/{yaml,front_preview,side_by_side}/` plus `summary.csv`, which
records success or the exact rendering error per input image. `--limit N` smoke-tests first.

---

## 5. Cluster

**Condor** (`runners/condor/`) — each job has a `submit_*.sh` wrapper, a `.sub` file, and a
`run_*.sh` payload; the second positional argument is usually the bid:

```bash
bash runners/condor/submit_h100.sh smplx_grouped grouped 150      # DINOv2 pipeline
bash runners/condor/submit_garment_tree_balanced_h100.sh garment_tree_v1 150
bash runners/condor/submit_garment_tree_ensemble_h100.sh ensemble_v1 150
bash runners/condor/submit_chatgarment_smplx_h100.sh smplx_vlm_lora 150
```

Jobs request one 80 GB H100 and print their log and run directories. W&B credentials go in
`runners/condor/secrets.sh` (git-ignored, `chmod 600`).

**Slurm** (`runners/slurm/batch/*.sbatch`) covers the DINOv2 train/infer variants.

If `images.json` was written on another machine, rewrite the stored prefix at load time:

```bash
--image-path-prefix /old/absolute/root /new/absolute/root
```

---

## 6. Gotchas & invariants

* **Split by garment/body id, never by image row.** All views and poses of a garment share
  identical targets, so an image-level split leaks.
* **The mask is a deterministic function of the categoricals**, not independent information —
  a slot is active only because some type selector turned that branch on. For an unlabeled
  image, derive the mask from the *predicted* categoricals.
* **`const_ranges` are empirical** (per-slot min/max over the records a run saw). A `--limit`
  run or a different file set yields different ranges — always decode with the same
  `schema.json` you trained against. The `[SEG]` 0–1 values are run-independent.
* **The schema is a union over all records**, including garments whose images you don't have,
  so slot indices stay stable across partial downloads. Absent families stay permanently
  mask-0.
* **Split-record ownership (ChatGarment):** in records split into `upperbody_garment` +
  `lowerbody_garment`, each half is an independently sampled complete outfit, so the upper
  half carries a throwaway bottom and vice versa. `is_owned_path` keeps only
  `shirt/collar/sleeve/left` (+`meta.upper`) from the upper half and the skirt/pants groups +
  `waistband` (+`meta.wb`, `meta.bottom`) from the lower half. Naive merging injects garbage.
* **Robustness:** unparseable records and `#[SEG] ≠ len(all_floats)` mismatches are warned and
  skipped per-gid, never fatal.
* **Legacy targets** `shirt.openfront` and `waistband.height` exist in prepared data but are
  excluded from new models by default (`--include-unsupported-params` restores them).

### Reference run (ChatGarment v2, frame 0, full file)

```
[scan] records=350256 parsed_gids=9476 parse_fail=0 seg_mismatch=0
[scan] images found=4272 missing=33632 frame_filtered=312352
[schema] continuous slots: 152   fixed-constant slots: 32   categorical fields: 60
[keep] garments with valid config + local images: 1068
[split] train=842 val=113 test=113 (seed=42)
```

---

## 7. Tests

```bash
pytest tests/
```

`tests/test_garment_tree_model.py` exercises GarmentTreeNet shapes against a tiny schema;
`tests/test_garment_ensemble_routes.py` checks route-constraint construction, route marginal
recovery, and expert-head region routing.

---

## 8. File map

| File | Role |
| --- | --- |
| `prepare_data.py` | ChatGarment preparer + `GarmentDataset` |
| `prepare_garmentcode_smplx.py` | GarmentCodeSMPLX preparer (body-disjoint splits) |
| `compile_balanced_garmentcode_smplx.py` | category-quota balanced compiler |
| `compile_output_balanced_garmentcode_smplx.py` | output-space balanced compiler (keeps all garments) |
| `compile_chatgarment_auxiliary.py` | map ChatGarment data onto an existing schema |
| `report_balanced_dataset_requirements.py` | explain rejected folders and quota deficits |
| `dinov2_pipeline.py` | train/infer entry point for the three DINOv2 architectures |
| `train_dinov2.py` / `model.py` / `train_dinov2_grouped.py` | baseline / per-parameter / grouped heads |
| `garment_tree_model.py`, `train_garment_tree.py` | GarmentTreeNet |
| `garment_ensemble_model.py`, `train_garment_ensemble.py` | route-constrained ensemble |
| `infer_dinov2.py`, `infer_garment_tree.py` | single-image inference (tree/ensemble auto-detected) |
| `edit_model.py`, `prepare_edit_data.py`, `train_edit_model.py`, `infer_edit.py`, `eval_edit.py`, `edit_to_design.py` | instruction-conditioned editing |
| `mesh_fitting_gnn.py`, `train_mesh_fitting_gnn.py`, `infer_mesh_fitting_gnn.py` | template-deformation GNN |
| `prediction_to_yaml.py` | decode predictions into a GarmentCode design YAML |
| `render_garmentcode.py` | headless generate + drape + render |
| `scripts/` | data fetching, batch comparison, plotting, cluster setup |
| `runners/` | Condor and Slurm job definitions |
