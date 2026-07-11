#!/usr/bin/env python3
"""verify_dump.py -- human-readable sanity dump of the ChatGarment parsing.

This is the *first part* of the pipeline only (no PyTorch / no fixed-slot
union).  For a handful of garments whose rendered images actually exist under
``data/chatgarment_data/garments_imgs_v2_3``, it writes a JSON file recording,
per garment id:

  * where the data comes from  -- which ``data_img_v*.json`` the config was read
    from, the original cluster image path, and the local image files on disk,
  * the data itself            -- the decoded continuous targets (each ``[SEG]``
    float paired with its dotted key-path, in document order), the categorical
    targets, the merged garment meta, and the raw config string so the
    ``[SEG]`` -> float mapping can be eyeballed.

The local folder ``garments_imgs_v2_3`` merges renders from both v2 and v3, and
the two versions reuse gid numbers for *different* garments.  To keep the
image<->config correspondence unambiguous, we only dump garments whose gid
appears in exactly one of the two JSON files.
"""

import argparse
import json
import os

from prepare_data import (
    iter_records,
    parse_image_path,
    parse_config,
    walk_config,
    extract_meta,
)

IMG_ROOT = "data/chatgarment_data/garments_imgs_v2_3"
JSONS = {
    "data_img_v2.json": "data/chatgarment_data/training/synthetic/data_img_v2.json",
    "data_img_v3.json": "data/chatgarment_data/training/synthetic/data_img_v3.json",
}


def flatten(all_floats):
    out = []
    for x in all_floats:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def local_gids():
    return {
        d for d in os.listdir(IMG_ROOT)
        if d.isdigit() and os.path.isdir(os.path.join(IMG_ROOT, d))
    }


def gid_to_source(want):
    """For each wanted gid, find which JSON(s) contain it. Returns
    gid -> {json_name: representative_record}."""
    hits = {g: {} for g in want}
    for name, path in JSONS.items():
        for rec in iter_records(path):
            pp = parse_image_path(rec.get("image", ""))
            if pp is None:
                continue
            gid = pp[0]
            if gid in hits and name not in hits[gid]:
                hits[gid][name] = rec
    return hits


def scan_local_frames(gid):
    """frame_str -> [view0..view3 local paths or None] for one gid folder."""
    base = os.path.join(IMG_ROOT, gid, "motion_0", "imgs")
    frames = {}
    if not os.path.isdir(base):
        return frames
    for frame in sorted(os.listdir(base), key=lambda s: int(s) if s.isdigit() else s):
        img_dir = os.path.join(base, frame, "img")
        if not os.path.isdir(img_dir):
            continue
        views = [None, None, None, None]
        for fn in os.listdir(img_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() == ".png" and stem.isdigit() and int(stem) < 4:
                views[int(stem)] = os.path.join(img_dir, fn)
        frames[frame] = views
    return frames


def decode(rec):
    gpt = next(c["value"] for c in rec["conversations"] if c["from"] == "gpt")
    config = parse_config(gpt)
    seg_paths, cat_items, const_items = [], [], []
    walk_config(config, [], seg_paths, cat_items, const_items)
    floats = flatten(rec.get("all_floats", []))
    meta = extract_meta(config)
    return gpt, seg_paths, cat_items, const_items, floats, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of garments to dump")
    ap.add_argument("--out", default="verify_dump/verify_garments.json")
    args = ap.parse_args()

    loc = local_gids()
    print(f"[info] {len(loc)} garment folders present under {IMG_ROOT}")

    hits = gid_to_source(loc)
    # Unambiguous = present in exactly one JSON.
    unambiguous = sorted(
        (g for g in loc if len(hits[g]) == 1), key=lambda s: int(s)
    )
    print(f"[info] {len(unambiguous)} of them appear in exactly one JSON "
          f"(unambiguous image<->config link)")

    chosen = unambiguous[: args.n]
    print(f"[info] dumping {len(chosen)} garments: {chosen}")

    dump = {}
    for gid in chosen:
        src_name, rec = next(iter(hits[gid].items()))
        gpt, seg_paths, cat_items, const_items, floats, meta = decode(rec)

        assert len(seg_paths) == len(floats), (
            f"gid {gid}: {len(seg_paths)} [SEG] vs {len(floats)} floats"
        )
        frames = scan_local_frames(gid)
        n_imgs = sum(1 for v in frames.values() for p in v if p)

        dump[f"{src_name.split('_')[-1].split('.')[0]}_{gid}"] = {
            "gid": gid,
            "source_json": src_name,
            "cluster_image_example": rec.get("image"),
            "meta": {"upper": meta[0], "wb": meta[1], "bottom": meta[2]},
            "n_local_images": n_imgs,
            "n_frames": len(frames),
            "n_continuous": len(seg_paths),
            "n_categorical": len(cat_items),
            "n_fixed_constants": len(const_items),
            "continuous_targets": [
                {"key_path": p, "value": v} for p, v in zip(seg_paths, floats)
            ],
            "categorical_targets": [
                {"key_path": p, "value": v} for p, v in cat_items
            ],
            # Plain numbers in the config that are NOT [SEG] targets in the
            # original ChatGarment protocol -- extracted anyway so they're
            # available if you later decide to train on them too.
            "fixed_constants": [
                {"key_path": p, "value": v} for p, v in const_items
            ],
            "local_frames": frames,
            "raw_gpt_config": gpt,
        }
        print(f"  gid {gid:>5} [{src_name}] meta={meta}  "
              f"cont={len(seg_paths)} cat={len(cat_items)} const={len(const_items)} "
              f"imgs={n_imgs}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[done] wrote {len(dump)} garments to {args.out}")


if __name__ == "__main__":
    main()
