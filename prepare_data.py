#!/usr/bin/env python3
"""prepare_data.py

Turn raw ChatGarment reconstruction JSON files into a clean, frozen
train/val/test split ready for a PyTorch training loop.

The ChatGarment dataset (``sy000/ChatGarmentDataset``) stores one row per
*image* in ``training/synthetic/data_img_v*.json``.  Every row carries:

  * ``image``       -- absolute path on the authors' compute cluster,
  * ``conversations`` -- a human prompt and a ``gpt`` answer.  The answer is a
                       Python-repr dict (single quotes, ``null``/``true``/
                       ``false``) whose ``[SEG]`` tokens mark every float that
                       must be predicted,
  * ``all_floats``  -- the (nested) list of those float values,

Many rows share the same garment: a garment ``<gid>`` is rendered at several
body-pose frames, each from 4 camera views, so hundreds of rows point at the
same targets.  This script:

  1. walks *every* record to build a **fixed slot layout** -- the union of all
     ``[SEG]`` key-paths (continuous targets), the union of all plain-number
     key-paths (fixed constants -- not a ``[SEG]`` target in the original
     ChatGarment protocol, but extracted anyway so they're available if you
     later decide to use them), and a per-key-path vocabulary of categorical
     values,
  2. remaps cluster image paths onto locally extracted ``--image-root``
     directories, keeping only rows whose file exists on disk and whose pose
     frame passes ``--frames`` (default: frame ``0`` only),
  3. groups surviving images by ``(garment_id, frame)`` -> 4 view paths,
  4. writes a garment-id-level, meta-stratified, seed-frozen split.

Outputs (written to ``--out``): ``schema.json``, ``targets.npz``,
``images.json``, ``splits.json``.  A ``GarmentDataset`` for the training loop
lives at the bottom of this file.

Only depends on numpy, Pillow, torch, torchvision -- no HuggingFace.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
# Parsing the ChatGarment record format
# --------------------------------------------------------------------------- #

# A placeholder that (a) is a valid Python string literal so ``ast.literal_eval``
# accepts it and (b) cannot collide with any real config value.
_SEG_SENTINEL = "__CHATGARMENT_SEG__"

# ``null`` / ``true`` / ``false`` only ever appear as *values* in these configs
# (keys and string values are single-quoted), so a token-boundary replace that
# refuses to fire when adjacent to a word char or quote is safe.
_NULL_RE = re.compile(r"(?<![\w'\"])null(?![\w'\"])")
_TRUE_RE = re.compile(r"(?<![\w'\"])true(?![\w'\"])")
_FALSE_RE = re.compile(r"(?<![\w'\"])false(?![\w'\"])")

# The segment immediately before ``motion_<N>`` in the image path is the gid.
_MOTION_RE = re.compile(r"^motion_\d+$")


def parse_config(gpt_value: str):
    """Parse a ``gpt`` answer string into a Python dict.

    Substitutes ``[SEG]`` for a sentinel string and the JSON keywords for their
    Python equivalents, then ``ast.literal_eval``.  Raises on malformed input;
    callers are expected to catch and skip.
    """
    s = gpt_value.replace("[SEG]", "'%s'" % _SEG_SENTINEL)
    s = _NULL_RE.sub("None", s)
    s = _TRUE_RE.sub("True", s)
    s = _FALSE_RE.sub("False", s)
    return ast.literal_eval(s)


def _is_pathlike(val) -> bool:
    """True for free-form asset paths that must never become categoricals."""
    if not isinstance(val, str):
        return False
    if "\\" in val or "/" in val:
        return True
    return val.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".obj"))


def walk_config(node, path, seg_paths, cat_items, const_items):
    """Recursively walk a parsed config in document order.

    Appends dotted key-paths of every ``[SEG]`` sentinel to ``seg_paths`` (in
    the order they appear, matching ``all_floats``), ``(key_path, value)`` for
    every categorical leaf (str / bool / None that is not a SEG sentinel and
    not a file path) to ``cat_items``, and ``(key_path, value)`` for every
    plain-number leaf (a "fixed constant" -- not a ``[SEG]`` prediction target
    in the original ChatGarment protocol, but a real per-garment value that
    varies across records) to ``const_items``.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            walk_config(v, path + [str(k)], seg_paths, cat_items, const_items)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk_config(v, path + [str(i)], seg_paths, cat_items, const_items)
    elif node == _SEG_SENTINEL:
        seg_paths.append(".".join(path))
    elif isinstance(node, (str, bool)) or node is None:
        # ``isinstance(True, int)`` is True, so bool is caught here (before the
        # numeric branch) and correctly treated as categorical.
        if not _is_pathlike(node):
            cat_items.append((".".join(path), node))
    else:  # plain int/float -> fixed constant (not a [SEG] target, but real data)
        const_items.append((".".join(path), float(node)))


# --------------------------------------------------------------------------- #
# Ownership of parameter groups in split (upperbody/lowerbody) records.
#
# Each half of a split record is a *complete, independently-sampled* GarmentCode
# design: upperbody_garment carries its own throwaway bottom (meta.bottom +
# skirt/pants params that were never rendered), and lowerbody_garment carries a
# throwaway top. Only the top of the upper half and the bottom (+waistband) of
# the lower half appear in the images. The discarded half's values are pure
# noise w.r.t. the image -- unlearnable -- so they are dropped here and never
# become schema slots or training targets (their would-be slots simply don't
# exist; anything sharing a key-path stays mask=0 / y_cat=-1 for this garment).
# Verified visually: gid v2_1327's render matches lowerbody's SkirtLevels, not
# upperbody's junk SkirtManyPanels.
# --------------------------------------------------------------------------- #

_UPPER_GROUPS = {"shirt", "collar", "sleeve", "left"}
_LOWER_GROUPS = {"skirt", "flare-skirt", "godet-skirt", "pencil-skirt",
                 "pants", "levels-skirt", "waistband"}


def is_owned_path(path):
    """True if this key-path belongs to the rendered half of its record.

    wholebody_garment records are a single self-consistent sample -- everything
    is owned. For split records: shirt/collar/sleeve/left (+meta.upper) belong
    to upperbody_garment; the skirt/pants groups and waistband (+meta.wb,
    meta.bottom) belong to lowerbody_garment.
    """
    parts = path.split(".")
    top = parts[0]
    group = parts[1] if len(parts) > 1 else ""
    if top == "upperbody_garment":
        if group == "meta":
            return len(parts) > 2 and parts[2] == "upper"
        return group in _UPPER_GROUPS
    if top == "lowerbody_garment":
        if group == "meta":
            return len(parts) > 2 and parts[2] in ("wb", "bottom")
        return group in _LOWER_GROUPS
    return True  # wholebody_garment (or anything unrecognised): keep


def extract_meta(config):
    """Merge the ``meta`` blocks of a config into one ``(upper, wb, bottom)``.

    Whole-body records have a single authoritative ``meta``. Split records have
    two, and each is a complete independent sample -- so ``upper`` is trusted
    only from upperbody_garment and ``wb``/``bottom`` only from
    lowerbody_garment (the same ownership rule as is_owned_path); the other
    half's value for that field is a discarded sample, not the rendered outfit.
    """
    owner = {"upper": "upperbody_garment",
             "wb": "lowerbody_garment",
             "bottom": "lowerbody_garment"}
    out = {"upper": None, "wb": None, "bottom": None}
    for top, sub in config.items():
        meta = sub.get("meta") if isinstance(sub, dict) else None
        if not isinstance(meta, dict):
            continue
        for f in out:
            v = meta.get(f)
            if v is not None and (top == "wholebody_garment" or top == owner[f]):
                out[f] = v
    return (out["upper"], out["wb"], out["bottom"])


def flatten_floats(all_floats):
    """Flatten ``all_floats``. Usually a list of lists, but a few records store
    a flat list of floats -- handle both."""
    out = []
    for x in all_floats:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def parse_image_path(image_path):
    """Return ``(gid, frame, view_idx, tail)`` from a cluster image path.

    ``tail`` is everything from the gid onward, e.g.
    ``<gid>/motion_0/imgs/<frame>/img/<view>.png`` -- used to re-root onto a
    local ``--image-root``.  Returns ``None`` if the path shape is unexpected.
    """
    parts = image_path.replace("\\", "/").split("/")
    m_idx = None
    for i, seg in enumerate(parts):
        if _MOTION_RE.match(seg):
            m_idx = i
            break
    if m_idx is None or m_idx == 0 or m_idx + 4 >= len(parts):
        return None
    gid = parts[m_idx - 1]
    frame = parts[m_idx + 2]
    view_name = parts[m_idx + 4]
    try:
        view_idx = int(os.path.splitext(view_name)[0])
    except ValueError:
        return None
    tail = "/".join(parts[m_idx - 1:])
    return gid, frame, view_idx, tail


# --------------------------------------------------------------------------- #
# Streaming reader (the JSON files are up to ~1.4 GB -- read once, no json.load
# of the whole thing into a second structure)
# --------------------------------------------------------------------------- #

def iter_records(json_path, limit=None):
    """Yield records from a top-level JSON array one at a time.

    Uses ``JSONDecoder.raw_decode`` over the file buffer so the array is decoded
    incrementally rather than materialising a second copy via ``json.loads`` of
    the entire text.
    """
    with open(json_path, "r") as f:
        buf = f.read()
    dec = json.JSONDecoder()
    i = buf.index("[") + 1
    n = 0
    length = len(buf)
    while i < length:
        while i < length and buf[i] in " ,\n\r\t":
            i += 1
        if i >= length or buf[i] == "]":
            break
        obj, i = dec.raw_decode(buf, i)
        yield obj
        n += 1
        if limit is not None and n >= limit:
            break


# --------------------------------------------------------------------------- #
# The single pass: schema union + image manifest, together
# --------------------------------------------------------------------------- #

def build(sets, limit, frames=None):
    """One pass per JSON file. Returns everything needed to write the outputs.

    ``sets`` is a list of ``(tag, json_path, image_roots)`` triples.  Each JSON
    is matched *only* against its own image roots: dataset versions (v2/v3/...)
    reuse the same garment-id numbers for different garments, so probing every
    root with every JSON would silently pair images with the wrong config.  The
    schema union is still built across all sets so one model can train on all
    versions with stable slot indices.

    ``frames`` optionally restricts the image manifest to a subset of pose
    frames (e.g. ``{"0"}`` to keep only the ``.../imgs/0/img/*.png`` renders).
    ``None`` keeps every frame. This only prunes *images*, never garments --
    the schema/config is still built from every record so slot indices stay
    stable regardless of which frames you happen to keep.

    Config is parsed once per garment id (all rows of a gid share it), while
    image existence is checked for every row.
    """
    # Per-gid cached parse: gid -> (seg_paths, floats, cat_dict, const_dict, meta_tuple)
    configs = {}
    bad_gids = set()
    # Union schema.
    union_seg = set()
    const_stats = {}  # path -> [min, max] over ALL records (for 0-1 normalisation)
    cat_vocab = defaultdict(set)
    # Image manifest: gid -> {frame -> {view_idx -> local_path}}
    images = defaultdict(lambda: defaultdict(dict))

    n_records = 0
    n_img_found = 0
    n_img_missing = 0
    n_img_frame_skipped = 0
    n_parse_fail = 0
    n_seg_mismatch = 0

    for tag, json_path, image_roots in sets:
        print(f"[scan] reading {json_path} (tag={tag!r}, roots={image_roots})",
              flush=True)
        for rec in iter_records(json_path, limit=limit):
            n_records += 1
            image_path = rec.get("image")
            if not image_path:
                continue
            parsed_path = parse_image_path(image_path)
            if parsed_path is None:
                continue
            gid_raw, frame, view_idx, tail = parsed_path
            gid = f"{tag}_{gid_raw}" if tag else gid_raw

            # ---- schema (once per gid) ----
            # Runs regardless of the --frames filter below: the config/floats are
            # identical across every frame of a gid, so restricting which frames
            # end up in the image manifest must never restrict which records
            # contribute to the schema union.
            if gid not in configs and gid not in bad_gids:
                try:
                    gpt = next(
                        c["value"] for c in rec["conversations"] if c["from"] == "gpt"
                    )
                    config = parse_config(gpt)
                except Exception as e:  # unparseable record -> skip this gid
                    n_parse_fail += 1
                    bad_gids.add(gid)
                    print(f"[warn] gid {gid}: parse failed ({e})", flush=True)
                    continue

                seg_paths = []
                cat_items = []
                const_items = []
                walk_config(config, [], seg_paths, cat_items, const_items)
                floats = flatten_floats(rec.get("all_floats", []))

                if len(seg_paths) != len(floats):
                    n_seg_mismatch += 1
                    bad_gids.add(gid)
                    print(
                        f"[warn] gid {gid}: {len(seg_paths)} [SEG] tokens vs "
                        f"{len(floats)} floats -- skipping",
                        flush=True,
                    )
                    continue

                # Ownership filter (only AFTER the [SEG]<->floats alignment
                # check above, which depends on document order): drop the
                # discarded half's throwaway values so they never become
                # schema slots or mask=1 training targets.
                owned = [(p, float(v)) for p, v in zip(seg_paths, floats)
                         if is_owned_path(p)]
                seg_paths = [p for p, _ in owned]
                seg_floats = [v for _, v in owned]
                cat_items = [(p, v) for p, v in cat_items if is_owned_path(p)]
                const_items = [(p, v) for p, v in const_items if is_owned_path(p)]

                cat_dict = {}
                for p, val in cat_items:
                    cat_dict[p] = val  # last write wins (paths are unique anyway)
                    cat_vocab[p].add(val)
                const_dict = dict(const_items)  # paths are unique within one config
                union_seg.update(seg_paths)
                for p, v in const_dict.items():
                    s = const_stats.get(p)
                    if s is None:
                        const_stats[p] = [v, v]
                    else:
                        if v < s[0]:
                            s[0] = v
                        if v > s[1]:
                            s[1] = v
                configs[gid] = (
                    seg_paths,
                    seg_floats,
                    cat_dict,
                    const_dict,
                    extract_meta(config),
                )

            # ---- image manifest (every row, subject to the --frames filter) ----
            if frames is not None and frame not in frames:
                n_img_frame_skipped += 1
                continue
            found = None
            for root in image_roots:
                cand = os.path.join(root, tail)
                if os.path.isfile(cand):
                    found = cand
                    break
            if found is None:
                n_img_missing += 1
                continue
            n_img_found += 1
            images[gid][frame][view_idx] = found

    print(
        f"[scan] records={n_records} parsed_gids={len(configs)} "
        f"parse_fail={n_parse_fail} seg_mismatch={n_seg_mismatch}",
        flush=True,
    )
    print(
        f"[scan] images found={n_img_found} missing={n_img_missing} "
        f"frame_filtered={n_img_frame_skipped}",
        flush=True,
    )

    return configs, union_seg, const_stats, cat_vocab, images


# --------------------------------------------------------------------------- #
# Finalise schema + encode targets + split
# --------------------------------------------------------------------------- #

def finalize_schema(union_seg, const_stats, cat_vocab):
    """Freeze the slot layout. Sorted key-paths -> stable indices regardless of
    which subset of records happened to be present locally."""
    cont_slots = OrderedDict((p, i) for i, p in enumerate(sorted(union_seg)))
    const_slots = OrderedDict((p, i) for i, p in enumerate(sorted(const_stats)))
    # Per-slot [min, max] over the whole dataset, so training code can put the
    # constants on the same 0-1 scale as the (pre-normalised) [SEG] floats and
    # decode back afterwards.
    const_ranges = OrderedDict((p, list(const_stats[p])) for p in const_slots)

    cat_final = OrderedDict()
    for p in sorted(cat_vocab):
        vals = [v for v in cat_vocab[p] if not _is_pathlike(v)]
        if not vals:
            continue  # nothing categorical survived (all file paths) -> drop
        # Sort with None/bool/str mixed: key on (type-rank, string form).
        vals_sorted = sorted(vals, key=lambda v: (v is not None, str(v)))
        cat_final[p] = vals_sorted
    return cont_slots, const_slots, const_ranges, cat_final


def encode_targets(gids, configs, cont_slots, const_slots, cat_final):
    """Build ``y_cont``, ``mask``, ``y_const``, ``const_mask`` and ``y_cat``
    matrices over ``gids``."""
    n_cont = len(cont_slots)
    n_const = len(const_slots)
    cat_paths = list(cat_final.keys())
    cat_index = {p: {v: i for i, v in enumerate(cat_final[p])} for p in cat_paths}
    n_cat = len(cat_paths)

    y_cont = np.zeros((len(gids), n_cont), dtype=np.float32)
    mask = np.zeros((len(gids), n_cont), dtype=np.float32)
    y_const = np.zeros((len(gids), n_const), dtype=np.float32)
    const_mask = np.zeros((len(gids), n_const), dtype=np.float32)
    y_cat = np.full((len(gids), n_cat), -1, dtype=np.int64)

    for row, gid in enumerate(gids):
        seg_paths, floats, cat_dict, const_dict, _meta = configs[gid]
        for path, val in zip(seg_paths, floats):
            slot = cont_slots.get(path)
            if slot is not None:
                y_cont[row, slot] = val
                mask[row, slot] = 1.0
        for path, val in const_dict.items():
            slot = const_slots.get(path)
            if slot is not None:
                y_const[row, slot] = val
                const_mask[row, slot] = 1.0
        for c, path in enumerate(cat_paths):
            if path in cat_dict:
                idx = cat_index[path].get(cat_dict[path])
                if idx is not None:
                    y_cat[row, c] = idx
    return y_cont, mask, y_const, const_mask, y_cat


def make_splits(gids, configs, val_ratio, test_ratio, seed):
    """Garment-id-level split, stratified by ``(upper, wb, bottom)`` signature,
    frozen with ``seed``."""
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for gid in gids:
        buckets[configs[gid][4]].append(gid)

    train, val, test = [], [], []
    for sig in sorted(buckets, key=lambda s: tuple("" if x is None else str(x) for x in s)):
        members = sorted(buckets[sig])
        rng.shuffle(members)
        n = len(members)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        # Guarantee train is never emptied out for tiny buckets.
        n_val = min(n_val, max(0, n - n_test - 1)) if n - n_test >= 1 else 0
        test += members[:n_test]
        val += members[n_test:n_test + n_val]
        train += members[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+",
                    help="one or more data_img_v*.json files (single-version mode; "
                         "all share --tag and --image-root)")
    ap.add_argument("--image-root", nargs="+",
                    help="local dirs holding extracted <gid>/motion_*/... "
                         "(single-version mode)")
    ap.add_argument("--tag", default="",
                    help="namespace prefix for garment ids (single-version mode)")
    ap.add_argument("--set", dest="sets", action="append", nargs="+",
                    metavar="TAG JSON ROOT",
                    help="multi-version mode: TAG JSON ROOT [ROOT...], repeatable. "
                         "Each JSON is matched only against its own roots -- "
                         "required when combining versions, because v2/v3/... "
                         "reuse garment-id numbers for different garments. "
                         "e.g. --set v2 data_img_v2.json imgs_v2_1 imgs_v2_3 "
                         "--set v3 data_img_v3.json imgs_v3")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--val", type=float, default=0.1, help="validation fraction")
    ap.add_argument("--test", type=float, default=0.1, help="test fraction")
    ap.add_argument("--seed", type=int, default=42, help="split seed (frozen)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N records per file (smoke test)")
    ap.add_argument("--frames", nargs="+", default=["0"],
                    help="only keep images from these pose frames (folder names "
                         "under motion_0/imgs/, e.g. '0' or '0 30 60'). Does not "
                         "affect the schema, which always covers every record. "
                         "Pass --frames all to keep every frame.")
    args = ap.parse_args()

    frame_filter = None if args.frames == ["all"] else set(args.frames)

    # Assemble (tag, json, roots) sets from either interface.
    sets = []
    if args.sets:
        if args.json or args.image_root:
            ap.error("use either --set or --json/--image-root, not both")
        for entry in args.sets:
            if len(entry) < 3:
                ap.error(f"--set needs TAG JSON ROOT [ROOT...], got: {entry}")
            tag, jp, roots = entry[0], entry[1], entry[2:]
            sets.append((tag, jp, [os.path.abspath(r) for r in roots]))
        tags = [t for t, _, _ in sets]
        if len(set(tags)) != len(tags):
            ap.error(f"--set tags must be unique, got: {tags}")
    else:
        if not args.json or not args.image_root:
            ap.error("provide --json and --image-root, or repeatable --set")
        roots = [os.path.abspath(r) for r in args.image_root]
        sets = [(args.tag, jp, roots) for jp in args.json]

    for _, jp, roots in sets:
        if not os.path.isfile(jp):
            ap.error(f"json file not found: {jp}")
        for r in roots:
            if not os.path.isdir(r):
                print(f"[warn] image root does not exist: {r}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    print(f"[frames] keeping: {'all' if frame_filter is None else sorted(frame_filter, key=int)}",
          flush=True)
    configs, union_seg, const_stats, cat_vocab, images = build(
        sets, args.limit, frames=frame_filter
    )

    cont_slots, const_slots, const_ranges, cat_final = finalize_schema(
        union_seg, const_stats, cat_vocab
    )
    print(f"[schema] continuous slots (union floats, [SEG] targets): {len(cont_slots)}", flush=True)
    print(f"[schema] fixed-constant slots (not [SEG], extracted for later use): {len(const_slots)}", flush=True)
    print(f"[schema] categorical fields: {len(cat_final)}", flush=True)

    # A garment is kept only if it has a valid config AND at least one local image.
    kept = sorted(g for g in images if g in configs and any(images[g].values()))
    print(f"[keep] garments with valid config + local images: {len(kept)}", flush=True)
    if not kept:
        print("[error] no garments matched local images -- check --image-root", flush=True)
        sys.exit(1)

    # ---- targets.npz ----
    y_cont, mask, y_const, const_mask, y_cat = encode_targets(
        kept, configs, cont_slots, const_slots, cat_final
    )
    np.savez(
        os.path.join(args.out, "targets.npz"),
        gids=np.array(kept, dtype=object),
        y_cont=y_cont,
        mask=mask,
        y_const=y_const,
        const_mask=const_mask,
        y_cat=y_cat,
    )

    # ---- schema.json ----
    with open(os.path.join(args.out, "schema.json"), "w") as f:
        json.dump(
            {
                "cont_slots": cont_slots,
                "const_slots": const_slots,
                "const_ranges": const_ranges,
                "cat_vocab": cat_final,
                "n_cont": len(cont_slots),
                "n_const": len(const_slots),
                "n_cat": len(cat_final),
            },
            f,
            indent=1,
        )

    # ---- images.json ----
    images_out = {}
    for gid in kept:
        frames = {}
        for frame, views in images[gid].items():
            if not views:
                continue
            frames[frame] = [views.get(v) for v in range(4)]
        if frames:
            images_out[gid] = {
                "meta": list(configs[gid][4]),
                "frames": frames,
            }
    with open(os.path.join(args.out, "images.json"), "w") as f:
        json.dump(images_out, f)

    # ---- splits.json ----
    split_gids = sorted(images_out.keys())
    train, val, test = make_splits(split_gids, configs, args.val, args.test, args.seed)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(
            {
                "train": train,
                "val": val,
                "test": test,
                "seed": args.seed,
                "ratios": [1.0 - args.val - args.test, args.val, args.test],
            },
            f,
            indent=1,
        )

    print(
        f"[split] train={len(train)} val={len(val)} test={len(test)} "
        f"(seed={args.seed})",
        flush=True,
    )
    print(f"[done] wrote schema.json, targets.npz, images.json, splits.json to {args.out}",
          flush=True)


# --------------------------------------------------------------------------- #
# GarmentDataset
# --------------------------------------------------------------------------- #

import torch  # noqa: E402  (kept below the CLI so `--help` needs no torch)


class PadToSquare:
    """Pad a portrait or landscape image to a square without stretching it."""

    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, image):
        from PIL import Image

        side = max(image.width, image.height)
        canvas = Image.new("RGB", (side, side), self.fill)
        left = (side - image.width) // 2
        top = (side - image.height) // 2
        canvas.paste(image.convert("RGB"), (left, top))
        return canvas


class GarmentDataset(torch.utils.data.Dataset):
    """PyTorch dataset over a prepared split.

    ``single`` mode has one item per garment and samples one frame-0 view.
    ``all_images`` mode expands each garment into one item per available frame-0
    image, while preserving the garment-level train/validation/test split.

    Parameters
    ----------
    prepared_dir : str
        Directory produced by this script (schema/targets/images/splits).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    mode : str
        ``"single"`` -> one item per garment; ``"all_images"`` -> one item per
        available frame-0 image. Both return one image tensor ``[C, H, W]``.
    train : bool
        In ``single`` mode, ``True`` selects a random frame-0 view each time;
        ``False`` deterministically uses view 0 (or the first available view).
        ``all_images`` mode always enumerates every available frame-0 image.
    image_size : int
        Square resize applied by the default transform.
    transform : callable, optional
        Torchvision-style transform mapping a PIL image to a tensor. Defaults
        to light train augmentation or deterministic resize for evaluation.
    """

    def __init__(self, prepared_dir, split, mode="single", train=True,
                 image_size=224, transform=None, augmentation="light",
                 aspect_pad=False):
        if mode not in ("single", "all_images"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'single' or 'all_images'")
        self.prepared_dir = prepared_dir
        self.split = split
        self.mode = mode
        self.train = train

        with open(os.path.join(prepared_dir, "splits.json")) as f:
            splits = json.load(f)
        with open(os.path.join(prepared_dir, "images.json")) as f:
            self.images = json.load(f)
        with open(os.path.join(prepared_dir, "schema.json")) as f:
            self.schema = json.load(f)

        npz = np.load(os.path.join(prepared_dir, "targets.npz"), allow_pickle=True)
        gid_list = [str(g) for g in npz["gids"].tolist()]
        self.row_of = {g: i for i, g in enumerate(gid_list)}
        self.y_cont = npz["y_cont"]
        self.mask = npz["mask"]
        self.const_mask = npz["const_mask"]
        self.y_cat = npz["y_cat"]

        # Constants are stored raw (degrees, counts, cm). Normalise each slot to
        # 0-1 with its schema range so they live on the same scale as y_cont and
        # can be regressed by the same head. Decode: raw = lo + pred * (hi - lo),
        # then round (they sit on integer grids).
        self.y_const_raw = npz["y_const"]
        ranges = np.array(list(self.schema["const_ranges"].values()),
                          dtype=np.float32)          # [n_const, 2]
        if ranges.size:
            ranges = ranges.reshape(-1, 2)
            lo, hi = ranges[:, 0], ranges[:, 1]
            span = np.maximum(hi - lo, 1e-8)
            self.y_const = ((self.y_const_raw - lo) / span).astype(np.float32)
            self.y_const *= self.const_mask            # keep inactive slots at 0
        else:
            lo = hi = np.zeros((0,), dtype=np.float32)
            self.y_const = self.y_const_raw.astype(np.float32, copy=False)
        self.const_lo, self.const_hi = lo, hi

        # Keep only garments present in every required structure.
        self.garment_ids = [
            g for g in splits[split]
            if g in self.images and g in self.row_of
        ]
        self.image_samples = []
        self.refresh_samples()

        if augmentation not in ("none", "light", "domain"):
            raise ValueError(
                f"Unknown augmentation {augmentation!r}; "
                "expected 'none', 'light', or 'domain'"
            )

        if transform is None:
            from torchvision import transforms
            square_prefix = [PadToSquare()] if aspect_pad else []
            if train and augmentation == "light":
                # Keep geometry changes mild: garment labels can encode small
                # shape details, and horizontal flips may change semantics.
                self.transform = transforms.Compose([
                    *square_prefix,
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.88, 1.0),
                        ratio=(0.92, 1.08),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomApply([
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.10,
                            hue=0.02,
                        )
                    ], p=0.8),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                    ], p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(
                        p=0.1,
                        scale=(0.02, 0.08),
                        ratio=(0.5, 2.0),
                        value="random",
                    ),
                ])
            elif train and augmentation == "domain":
                self.transform = transforms.Compose([
                    *square_prefix,
                    transforms.RandomAffine(
                        degrees=7.0,
                        translate=(0.05, 0.05),
                        scale=(0.90, 1.08),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        fill=(255, 255, 255),
                    ),
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomApply([
                        transforms.ColorJitter(
                            brightness=0.35,
                            contrast=0.30,
                            saturation=0.25,
                            hue=0.04,
                        )
                    ], p=0.9),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
                    ], p=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(
                        p=0.20,
                        scale=(0.02, 0.12),
                        ratio=(0.4, 2.5),
                        value="random",
                    ),
                ])
            else:
                self.transform = transforms.Compose([
                    *square_prefix,
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform

    def __len__(self):
        if self.mode == "all_images":
            return len(self.image_samples)
        return len(self.garment_ids)

    # -- helpers ----------------------------------------------------------- #
    def _load_image(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _pick_frame(self, frames):
        """Return only the frame-0 renders used by this training setup."""
        if "0" not in frames:
            available = ", ".join(sorted(frames, key=lambda key: int(key)))
            raise KeyError(
                "Required frame '0' is missing from the image manifest "
                f"(available frames: {available or 'none'}). Rerun prepare_data.py "
                "with --frames 0."
            )
        return "0", frames["0"]

    def refresh_samples(self):
        """Rebuild the per-image index after manifest path filtering/rewrites."""
        self.image_samples = []
        if self.mode != "all_images":
            return
        for gid in self.garment_ids:
            _, views = self._pick_frame(self.images[gid]["frames"])
            self.image_samples.extend(
                (gid, path) for path in views if path is not None
            )

    # -- protocol ---------------------------------------------------------- #
    def __getitem__(self, idx):
        if self.mode == "all_images":
            gid, path = self.image_samples[idx]
            image = self._load_image(path)
        else:
            gid = self.garment_ids[idx]
            _, views = self._pick_frame(self.images[gid]["frames"])
            present = [v for v in views if v is not None]
            if self.train:
                path = present[torch.randint(len(present), (1,)).item()]
            else:
                path = views[0] if views[0] is not None else present[0]
            image = self._load_image(path)

        row = self.row_of[gid]
        return {
            "gid": gid,
            "image": image,
            "y_cont": torch.from_numpy(self.y_cont[row].copy()),
            "mask": torch.from_numpy(self.mask[row].copy()),
            # Fixed constants, normalised to 0-1 via schema const_ranges so they
            # can be concatenated with y_cont into a single regression target
            # (152 + 32 = 184). Raw values are in self.y_const_raw.
            "y_const": torch.from_numpy(self.y_const[row].copy()),
            "const_mask": torch.from_numpy(self.const_mask[row].copy()),
            "y_cat": torch.from_numpy(self.y_cat[row].copy()),
        }


if __name__ == "__main__":
    main()
