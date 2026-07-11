# ChatGarment Data Pipeline (`prepare_data.py`)

Turns the raw [ChatGarment dataset](https://huggingface.co/datasets/sy000/ChatGarmentDataset)
(`sy000/ChatGarmentDataset`) into a clean, frozen train/val/test split ready for a
PyTorch image → sewing-pattern model. One command produces four files and a
ready-to-use `GarmentDataset`.

---

## 1. What the raw data is

ChatGarment renders each procedurally-generated garment (a "gid") on a body at
several **pose frames** (`0, 30, 60, …`), each from **4 camera views**
(`000–003`). The reconstruction target for every image is a GarmentCode
_design config_.

The file `training/synthetic/data_img_v*.json` is a JSON array with **one row per
image**. Each row:

```jsonc
{
  "image": "/ps/.../hood_simulation_garmentcode_v2/1327/motion_0/imgs/0/img/000.png",
  "conversations": [
    {"from": "human", "value": "<image>\nCan you estimate the outfit sewing pattern code?"},
    {"from": "gpt",   "value": "{'upperbody_garment': {'meta': {...}, 'collar': {'width': [SEG], 'fc_angle': 88, ...}}}"}
  ],
  "all_floats": [[0.89, 0.30, ...]]   // the values behind the [SEG] tokens, in order
}
```

Format quirks the pipeline handles for you:

- The `gpt` value is a **Python-repr dict** (single quotes, `null`/`true`/`false`),
  _not_ JSON. Parsed with `ast.literal_eval` after keyword substitution.
- **`[SEG]`** marks each continuous value the model must regress. The i-th `[SEG]`
  in document order corresponds to the i-th entry of flattened `all_floats`. These
  floats are **already normalized to 0–1** by ChatGarment.
- Plain numbers with no `[SEG]` (e.g. `fc_angle: 88`) are **fixed constants** —
  real per-garment values ChatGarment chose not to make `[SEG]` targets.
- String / bool / `null` leaves are **categoricals**.
- All rows of one gid share the identical config and `all_floats`; only the
  `image` path differs.

### The split-garment "ownership" subtlety (important)

Some records are a single `wholebody_garment`. Others are split into
`upperbody_garment` **and** `lowerbody_garment`. In split records **each half is a
complete, independently-sampled outfit** — so `upperbody_garment` carries its own
_throwaway_ bottom (a skirt/pants that was generated but never rendered) and
`lowerbody_garment` carries a throwaway top. Naively merging the two halves
injects garbage (e.g. gid `v2_1327`'s upper half claims `SkirtManyPanels`, but the
rendered skirt is the lower half's `SkirtLevels`).

The pipeline resolves this with an **ownership filter** (`is_owned_path`): for
split records it keeps only `shirt/collar/sleeve/left` (+`meta.upper`) from the
upper half and the skirt/pants groups + `waistband` (+`meta.wb`, `meta.bottom`)
from the lower half. Discarded values never become schema slots or training
targets. (Verified visually — the reconstructed design for `v2_1327` renders in
GarmentCode as the correct strapless-top + fitted-waistband + levels-skirt.)

---

## 2. Prerequisites

**Python** 3.11, packages: `numpy`, `Pillow`, `torch`, `torchvision`. No
HuggingFace dependency. (`--help` and building the outputs need only numpy/Pillow;
`torch`/`torchvision` are imported for the `GarmentDataset` at the bottom of the
file.)

**Data on disk** — you must have already downloaded and extracted:

1. One or more `data_img_v*.json` files, e.g.
   `data/chatgarment_data/training/synthetic/data_img_v2.json`.
2. The matching extracted image folder(s), laid out as
   `<root>/<gid>/motion_0/imgs/<frame>/img/00[0-3].png`, e.g.
   `data/chatgarment_data/garments_imgs_v2_3/`.

> The script does **not** download or unzip anything — it reads local files only.

**⚠ gid numbers are reused across dataset versions.** `gid 1` in v2 is a _different
garment_ than `gid 1` in v3. So each JSON must be paired with **only its own**
image folders, and the pipeline can't auto-detect the pairing — you assert it (see
multi-version usage). Do **not** extract v2 and v3 zips into the same directory.

### Downloading the raw data from HuggingFace

The full repo is **hundreds of GB** — don't `snapshot_download` the whole thing.
Pull only the JSON(s) and the matching image zip(s) you actually need.

```bash
pip install -U "huggingface_hub[cli]"

# the small JSON files (a few hundred MB to ~1.4 GB each)
huggingface-cli download sy000/ChatGarmentDataset \
  training/synthetic/data_img_v2.json \
  --repo-type dataset --local-dir ./data/chatgarment_data

# the matching image zip (large -- see table below for sizes)
huggingface-cli download sy000/ChatGarmentDataset \
  garments_imgs_v2_3.zip \
  --repo-type dataset --local-dir ./data/chatgarment_data
```

Then extract in place, e.g.:

```bash
cd data/chatgarment_data && unzip garments_imgs_v2_3.zip -d garments_imgs_v2_3
```

| version        | JSON (`--json`)                                | image zip(s) (`--image-root` after unzip)                                    | zip size            |
| -------------- | ---------------------------------------------- | ---------------------------------------------------------------------------- | ------------------- |
| v2             | `training/synthetic/data_img_v2.json`          | `garments_imgs_v2_1.zip`, `garments_imgs_v2_2.zip`, `garments_imgs_v2_3.zip` | ~32 / ~32 / ~8 GB   |
| v3             | `training/synthetic/data_img_v3.json`          | `garments_imgs_v3.zip`                                                       | ~27 GB              |
| v4             | `training/synthetic/data_img_v4.json`          | `garments_imgs_v4.zip`                                                       | ~19 GB              |
| v1 (rest-pose) | `training/synthetic/data_restpose_img_v1.json` | `garments_imgs_v1_1.zip` … `garments_imgs_v1_5.zip`                          | ~32 GB × 4 + ~28 GB |

> **Only the `v2` ↔ `garments_imgs_v2_3` pairing above has been empirically
> verified** (on-disk frame sets match `data_img_v2.json`'s references for
> 1068/1069 local gids). The other JSON↔zip pairings follow the same naming
> convention but have not been individually re-verified — do the same frame-set
> sanity check before trusting a new pairing (see §7).

You don't need every shard of a version — `garments_imgs_v2_3.zip` alone is
enough to exercise the pipeline (it's what `prepared_v2` in this repo was built
from); add `v2_1`/`v2_2` later only if you want more garments covered.

---

## 3. Usage

### Single version (the common case)

```bash
python prepare_data.py \
  --json       ./data/chatgarment_data/training/synthetic/data_img_v2.json \
  --image-root ./data/chatgarment_data/garments_imgs_v2_3 \
  --tag        v2 \
  --out        ./prepared_v2
```

`--tag v2` namespaces output gids as `v2_1327` (keeps ids unique if you later add
other versions). Multiple `--json` / `--image-root` values are allowed and all
share the one `--tag`.

### Multiple versions (pair each JSON with its own roots)

```bash
python prepare_data.py \
  --set v2 ./.../data_img_v2.json  ./.../garments_imgs_v2_1 ./.../garments_imgs_v2_3 \
  --set v3 ./.../data_img_v3.json  ./.../garments_imgs_v3 \
  --out ./prepared_all
```

Each `--set TAG JSON ROOT [ROOT...]` is scanned in isolation — a JSON is only ever
matched against the roots you paired with it, which is what prevents cross-version
gid collisions. The schema is still built as the **union across all sets**, so one
model trains on every version with stable slot indices. Use `--set` **or**
`--json/--image-root`, not both.

### All flags

| flag                   | default       | meaning                                                                                        |
| ---------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `--json`               | —             | one or more`data_img_v*.json` (single-version mode)                                            |
| `--image-root`         | —             | local dirs holding`<gid>/motion_*/…` (single-version mode)                                     |
| `--tag`                | `""`          | gid namespace prefix                                                                           |
| `--set TAG JSON ROOT…` | —             | multi-version mode, repeatable                                                                 |
| `--out`                | _required_    | output directory                                                                               |
| `--frames`             | `0`           | pose frames to keep (`0`, or `0 30 60`, or `all`). **Images only — never affects the schema.** |
| `--val` / `--test`     | `0.1` / `0.1` | split fractions (rest is train)                                                                |
| `--seed`               | `42`          | frozen split seed                                                                              |
| `--limit N`            | —             | process only first N records/file (smoke test)                                                 |

`--frames` defaults to **frame `0` only**. Pass `--frames all` for every pose.

---

## 4. Outputs (written to `--out`)

### `schema.json` — the slot layout (the "meaning" of every index)

```jsonc
{
  "cont_slots":   { "upperbody_garment.collar.width": 0, ... },   // key_path -> index (152)
  "const_slots":  { "upperbody_garment.collar.fc_angle": 0, ... },// key_path -> index (32)
  "const_ranges": { "upperbody_garment.collar.fc_angle": [70, 110], ... }, // [min,max] per const
  "cat_vocab":    { "lowerbody_garment.meta.bottom": ["Pants","PencilSkirt",...], ... }, // (60)
  "n_cont": 152, "n_const": 32, "n_cat": 60
}
```

### `targets.npz` — ground truth, **one row per garment**

| array        | shape      | meaning                                                    |
| ------------ | ---------- | ---------------------------------------------------------- |
| `gids`       | `(N,)`     | garment id per row (the join key), e.g.`"v2_1327"`         |
| `y_cont`     | `(N, 152)` | `[SEG]` continuous values, already 0–1; `0` where inactive |
| `mask`       | `(N, 152)` | `1` = active for this garment, `0` = not applicable        |
| `y_const`    | `(N, 32)`  | fixed constants,**stored RAW** (degrees, counts, …)        |
| `const_mask` | `(N, 32)`  | `1` = active, `0` = not applicable                         |
| `y_cat`      | `(N, 60)`  | categorical class**index**; `-1` = not applicable          |

`mask` / `-1` exist because the schema is a **union** over all garment types: any
single garment only lights up the slots for its own structure (a pencil-skirt row
has all `pants.*` / `flare-skirt.*` slots at mask 0). See §7 for why the mask is a
deterministic function of the categoricals.

### `images.json` — where the photos are, per garment

```jsonc
{
  "v2_1327": {
    "meta": ["FittedShirt", "FittedWB", "SkirtLevels"],
    "frames": {
      "0": ["<view000.png>", "<view001.png>", "<view002.png>", "<view003.png>"],
    },
  },
}
```

### `splits.json` — frozen garment-id split

```jsonc
{ "train": [gids], "val": [gids], "test": [gids], "seed": 42, "ratios": [0.8,0.1,0.1] }
```

Split is **by garment id** (never by image row — that would leak a garment across
splits) and **stratified** by the `(upper, wb, bottom)` signature so rare garment
types appear in all three splits.

---

## 5. Using `GarmentDataset` in training

```python
from prepare_data import GarmentDataset
from torch.utils.data import DataLoader

train_ds = GarmentDataset("prepared_v2", split="train", mode="single", train=True)
val_ds   = GarmentDataset("prepared_v2", split="val",   mode="single", train=False)

loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)
batch  = next(iter(loader))
# batch["image"]      [B, 3, 224, 224]
# batch["y_cont"]     [B, 152]   0-1
# batch["mask"]       [B, 152]
# batch["y_const"]    [B, 32]    ALREADY normalized to 0-1 (see §6)
# batch["const_mask"] [B, 32]
# batch["y_cat"]      [B, 60]    class indices, -1 = ignore
# batch["gid"]        list[str]
```

- `mode="single"`: `__len__` is the number of garments. Training picks one random
  frame-0 view per garment per epoch; evaluation uses view 0 (or the first present
  view).
- `mode="all_images"`: the garment-level split is expanded into one item per
  available frame-0 image. Every image is therefore visited once per epoch, while
  all views of a garment remain in the same train/validation/test split.
- Frame `0` is required in both modes; no other pose folder is used.
- The default transform is Resize(224) → ToTensor → ImageNet normalize; pass your
  own `transform=` to override.

### Suggested model + loss (DINOv2 + two MLP heads)

`y_cont` (152) and the 0–1 `y_const` (32) live on the same scale, so a single
**regression head of width 184** covers both; a **classification head** covers the
60 categorical fields.

```python
y_reg = torch.cat([batch["y_cont"], batch["y_const"]], dim=-1)      # [B, 184]
m_reg = torch.cat([batch["mask"],   batch["const_mask"]], dim=-1)   # [B, 184]

loss_reg = ((pred_reg - y_reg)**2 * m_reg).sum() / m_reg.sum().clamp(min=1)

loss_cat, off = 0.0, 0
for k, vocab in enumerate(vocab_sizes):        # vocab_sizes from schema cat_vocab
    loss_cat += F.cross_entropy(pred_logits[:, off:off+vocab],
                                batch["y_cat"][:, k], ignore_index=-1)
    off += vocab

loss = loss_reg + lambda_cat * loss_cat
```

`mask=0` → zero gradient for that slot; `ignore_index=-1` skips inapplicable
categorical fields automatically.

### Training baseline

`train_dinov2.py` implements the DINOv2 + two-head baseline above. It loads the
prepared split through `GarmentDataset`, freezes DINOv2 by default, trains a
184-wide regression head (`y_cont` + normalized `y_const`) and a 199-wide
categorical head (60 fields), and writes `last.pt`, `best.pt`, `history.json`,
and `config.json`.

```bash
conda run -n project python train_dinov2.py \
  --prepared-dir prepared_v2 \
  --out-dir runs/dinov2_vits14 \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --amp
```

If `prepared_v2/images.json` was created on another machine, rewrite the stored
absolute image prefix at load time:

```bash
conda run -n project python train_dinov2.py \
  --prepared-dir prepared_v2 \
  --out-dir runs/dinov2_vits14 \
  --image-path-prefix \
    /Users/siddharth/Study/3dv_project/data/chatgarment_data/garments_imgs_v2_3 \
    /mnt/beegfs/home/stud136/3dv_project/data/chatgarment_data/garments_imgs_v2_3
```

Useful options:

- `--mode single` uses one sampled view per garment; `--mode all_images` uses every
  available frame-0 image as its own training/evaluation sample.
- `--unfreeze-backbone` fine-tunes DINOv2 instead of training only the heads.
- `--amp` enables CUDA mixed precision.
- `--max-train-batches 1 --max-val-batches 1` runs a quick smoke test.

---

## 6. Decoding predictions back to a GarmentCode design

At inference you convert the 0–1 predictions back to real values, per slot:

```
continuous ([SEG]):  raw = lo + pred * (hi - lo)         # lo,hi from GarmentCode default.yaml
constants:           raw = lo + pred * (hi - lo)         # lo,hi from schema.json const_ranges
categoricals:        value = cat_vocab[field][argmax(logits_field)]
```

then write into GarmentCode's design template (see `make_garmentcode_design.py`,
which does exactly this for GT verification) and upload to the GarmentCode GUI.

### ⚠ Do NOT blindly round all constants

Constants sit on integer grids **except two float-valued ones**:
`flare-skirt.skirt-many-panels.panel_curve` (range −0.35…0.45) under both
`lowerbody_garment` and `wholebody_garment`. Rounding those to integers corrupts
them. The correct rule is **round only `type: int` params**, leave `type: float`
alone — the type comes from GarmentCode's `assets/design_params/default.yaml`.
`make_garmentcode_design.py`'s `set_nested()` already casts by declared type and is
correct. **`schema.json` does not yet record this int/float type** — if you build a
standalone decoder off `schema.json` alone, add the type lookup from `default.yaml`
(or extend the schema to carry `const_types`).

---

## 7. Gotchas & invariants (read before trusting a run)

- **The mask is a deterministic function of the categoricals**, not independent
  information — a slot is active only because some type selector (e.g.
  `meta.bottom`, `sleeve.cuff.type`) turned that branch on. The model does **not**
  predict a mask. For training/eval on this dataset you already have the true mask
  in `targets.npz`. For a novel unlabeled image, derive the mask from the
  _predicted_ categoricals (or just fill the full 184-wide design template and let
  GarmentCode read only the reachable subtree).
- **`const_ranges` are empirical** (per-slot min/max over the records this run
  saw). They are stable for a fixed set of `--json` files with no `--limit`, but a
  `--limit` run or a different file set yields different ranges. **Always decode
  with the same `schema.json` you trained against.** (The `[SEG]` 0–1 values, by
  contrast, are fixed by ChatGarment and run-independent.)
- **The schema is a union over ALL records** in the JSON, even garments whose
  images you don't have locally — so slot indices stay stable across partial image
  downloads. Columns for garment types absent from your local images stay
  permanently inactive (all mask 0). Kept garments = valid config **and** ≥1 local
  image passing `--frames`.
- **Robustness:** unparseable records and `#[SEG] ≠ len(all_floats)` mismatches are
  warned and skipped per-gid, never fatal.

### Reference run (v2, frame 0 only, full file)

```
[scan] records=350256 parsed_gids=9476 parse_fail=0 seg_mismatch=0
[scan] images found=4272 missing=33632 frame_filtered=312352
[schema] continuous slots: 152   fixed-constant slots: 32   categorical fields: 60
[keep] garments with valid config + local images: 1068
[split] train=842 val=113 test=113 (seed=42)
```

---

## 8. File map

| file                         | role                                                            |
| ---------------------------- | --------------------------------------------------------------- |
| `prepare_data.py`            | this pipeline +`GarmentDataset`                                 |
| `make_garmentcode_design.py` | decode one gid's GT to a GarmentCode design yaml (verification) |
| `verify_dump.py`             | human-readable dump of parsed targets for eyeballing            |
