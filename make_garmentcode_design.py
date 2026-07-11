#!/usr/bin/env python3
"""Reconstruct one real garment's GarmentCode design.yaml from the prepared
ChatGarment data, denormalized to GarmentCode's actual parameter ranges, ready
to upload into the GarmentCode GUI ("Design" tab -> "Upload").

Standalone (no torch import) so it runs in the `garmentcode` conda env, which
has pyyaml but not torch. Duplicates the small parsing helpers from
prepare_data.py rather than importing it (that module imports torch at load
time).
"""
import ast
import copy
import json
import random
import re

import yaml

DEFAULT_YAML = "/Users/siddharth/Study/Garment_Code/assets/design_params/default.yaml"
DATA_JSON = "/Users/siddharth/Study/3dv_project/data/chatgarment_data/training/synthetic/data_img_v2.json"
IMAGES_JSON = "/Users/siddharth/Study/3dv_project/prepared_v2/images.json"

_SEG_SENTINEL = "__CHATGARMENT_SEG__"
_NULL_RE = re.compile(r"(?<![\w'\"])null(?![\w'\"])")
_TRUE_RE = re.compile(r"(?<![\w'\"])true(?![\w'\"])")
_FALSE_RE = re.compile(r"(?<![\w'\"])false(?![\w'\"])")
_MOTION_RE = re.compile(r"^motion_\d+$")
TOP_PREFIXES = ("wholebody_garment", "upperbody_garment", "lowerbody_garment")


def parse_config(gpt_value):
    s = gpt_value.replace("[SEG]", "'%s'" % _SEG_SENTINEL)
    s = _NULL_RE.sub("None", s)
    s = _TRUE_RE.sub("True", s)
    s = _FALSE_RE.sub("False", s)
    return ast.literal_eval(s)


def flatten_floats(all_floats):
    out = []
    for x in all_floats:
        out.extend(x) if isinstance(x, list) else out.append(x)
    return out


def parse_image_path(image_path):
    parts = image_path.replace("\\", "/").split("/")
    for i, seg in enumerate(parts):
        if _MOTION_RE.match(seg):
            return parts[i - 1]
    return None


def walk_config(node, path, seg_paths, cat_items, const_items):
    if isinstance(node, dict):
        for k, v in node.items():
            walk_config(v, path + [str(k)], seg_paths, cat_items, const_items)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk_config(v, path + [str(i)], seg_paths, cat_items, const_items)
    elif node == _SEG_SENTINEL:
        seg_paths.append(".".join(path))
    elif isinstance(node, (str, bool)) or node is None:
        cat_items.append((".".join(path), node))
    else:
        const_items.append((".".join(path), float(node)))


def strip_top_prefix(path):
    parts = path.split(".")
    if parts[0] in TOP_PREFIXES:
        parts = parts[1:]
    return ".".join(parts)


UPPER_GROUPS = {"shirt", "collar", "sleeve", "left"}
LOWER_GROUPS = {"skirt", "flare-skirt", "godet-skirt", "pencil-skirt", "pants", "levels-skirt"}


def owning_container(top_prefix, key):
    """Decide which half's value to trust for one (top_prefix, stripped_key).

    When a record is split into upperbody_garment / lowerbody_garment, EACH
    half is actually its own complete, independently-sampled GarmentCode
    design (own meta.upper/wb/bottom, own skirt/pants params etc.) -- only
    half of each is ever rendered. E.g. upperbody_garment can carry its own
    throwaway 'flare-skirt' params from a bottom type it rolled but never
    used. Naively merging both halves by path would silently let a discarded
    sample's values overwrite (or be overwritten by) the real ones. The rule:
    shirt/collar/sleeve/left group -> only trust upperbody_garment; the skirt/
    pants/etc. groups -> only trust lowerbody_garment; waistband -> prefer
    lowerbody_garment (falls back to upperbody_garment if lower lacks it);
    meta -> handled separately (per-field, see build_design()).
    """
    if top_prefix == "wholebody_garment":
        return top_prefix  # single self-consistent sample, no split to resolve
    group = key.split(".")[0]
    if group in UPPER_GROUPS:
        return "upperbody_garment"
    if group in LOWER_GROUPS:
        return "lowerbody_garment"
    if group == "waistband":
        return "lowerbody_garment"  # resolved with a fallback at call site
    return top_prefix  # meta handled separately; anything else: no conflict expected


def flatten_yaml_spec(node, path, out):
    if isinstance(node, dict) and "v" in node and "type" in node:
        out[".".join(path)] = node
    elif isinstance(node, dict):
        for k, v in node.items():
            flatten_yaml_spec(v, path + [k], out)


def set_nested(tree, dotted_path, value):
    """Write ``value`` into the template's ``v`` field, cast to the param's own
    declared ``type``. GarmentCode consumes these values directly (e.g.
    ``range(num_levels)``), so an int-typed param carrying a YAML float like
    ``2.0`` crashes pattern assembly -- this cast is what makes the exported
    yaml actually loadable by the GarmentCode GUI."""
    parts = dotted_path.split(".")
    node = tree
    for p in parts[:-1]:
        node = node[p]
    spec = node[parts[-1]]
    t = spec.get("type")
    if value is not None:
        if t == "int":
            value = int(round(float(value)))
        elif t == "float":
            value = float(value)
        elif t == "bool":
            value = bool(value)
    spec["v"] = value


def main():
    import sys
    random.seed()  # true randomness for gid pick
    forced_gid = sys.argv[1] if len(sys.argv) > 1 else None

    with open(DEFAULT_YAML) as f:
        default_design = yaml.safe_load(f)["design"]
    spec_flat = {}
    flatten_yaml_spec(default_design, [], spec_flat)
    print(f"[info] loaded {len(spec_flat)} parameter specs from default.yaml")

    with open(IMAGES_JSON) as f:
        images = json.load(f)
    gid = forced_gid if forced_gid else random.choice(list(images.keys()))
    raw_gid = gid.split("_", 1)[1]
    print(f"[info] picked gid = {gid!r} (raw folder = {raw_gid!r})")

    rec = None
    # Stream-scan for the first record whose image path has this raw gid.
    import time
    t0 = time.time()
    with open(DATA_JSON) as f:
        buf = f.read()
    dec = json.JSONDecoder()
    i = buf.index("[") + 1
    length = len(buf)
    while i < length:
        while i < length and buf[i] in " ,\n\r\t":
            i += 1
        if i >= length or buf[i] == "]":
            break
        obj, i = dec.raw_decode(buf, i)
        g = parse_image_path(obj.get("image", ""))
        if g == raw_gid:
            rec = obj
            break
    print(f"[info] scanned data_img_v2.json in {time.time()-t0:.1f}s, found={rec is not None}")
    if rec is None:
        raise SystemExit(f"gid {raw_gid} not found in {DATA_JSON}")

    gpt = next(c["value"] for c in rec["conversations"] if c["from"] == "gpt")
    config = parse_config(gpt)
    seg_paths, cat_items, const_items = [], [], []
    walk_config(config, [], seg_paths, cat_items, const_items)
    floats = flatten_floats(rec["all_floats"])
    assert len(seg_paths) == len(floats), "SEG/float count mismatch"

    is_split = "upperbody_garment" in config or "lowerbody_garment" in config

    def top_prefix_of(path):
        p = path.split(".")[0]
        return p if p in TOP_PREFIXES else None

    def denorm(path, norm_val):
        key = strip_top_prefix(path)
        spec = spec_flat.get(key)
        if spec is None:
            return key, None, "no spec found"
        lo, hi = spec["range"][0], spec["range"][-1]
        return key, round(lo + norm_val * (hi - lo), 4), None

    # Gather every (top_prefix, key, value) triple across cont/cat/const, keeping
    # meta.* separate since it needs field-by-field resolution, not a group filter.
    resolved, meta_candidates, unresolved = [], {}, []
    for path, norm_val in zip(seg_paths, floats):
        key, val, err = denorm(path, norm_val)
        (unresolved.append((path, err)) if err else
         resolved.append((top_prefix_of(path), key, val)))
    for path, val in cat_items + const_items:
        key = strip_top_prefix(path)
        val = round(val, 2) if isinstance(val, float) else val
        tp = top_prefix_of(path)
        if key.startswith("meta."):
            meta_candidates.setdefault(key, {})[tp] = val
        else:
            resolved.append((tp, key, val))

    design = copy.deepcopy(default_design)

    # meta: upperbody_garment owns 'upper', lowerbody_garment owns 'bottom' --
    # each half's OWN throwaway guess at the other field must never leak in.
    # 'wb' can legitimately be set by either half; prefer lowerbody_garment's
    # (waistband is physically attached to the bottom piece) with a fallback.
    if is_split:
        owner_by_field = {"meta.upper": "upperbody_garment", "meta.bottom": "lowerbody_garment"}
        for field in ("meta.upper", "meta.bottom", "meta.wb"):
            cands = meta_candidates.get(field, {})
            if field in owner_by_field and owner_by_field[field] in cands:
                v = cands[owner_by_field[field]]
            else:
                v = cands.get("lowerbody_garment")
                if v is None:
                    v = cands.get("upperbody_garment")
            set_nested(design, field, v)
    else:
        for field, cands in meta_candidates.items():
            set_nested(design, field, next(iter(cands.values())))

    # Everything else: apply the group-ownership filter so a discarded half's
    # leftover params (e.g. upperbody_garment's own unused skirt type) never
    # get merged into the final design.
    dropped = []
    for top_prefix, key, val in resolved:
        if is_split and top_prefix is not None:
            owner = owning_container(top_prefix, key)
            if owner != top_prefix:
                dropped.append((f"{top_prefix}.{key}", val, f"discarded, owned by {owner}"))
                continue
        try:
            set_nested(design, key, val)
        except KeyError:
            unresolved.append((f"{top_prefix}.{key}" if top_prefix else key, "path not in default tree"))

    if unresolved:
        print(f"[warn] {len(unresolved)} paths could not be placed:")
        for p, why in unresolved:
            print(f"   {p}: {why}")
    print(f"[info] {len(dropped)} throwaway (non-owning-half) values discarded, e.g.:")
    for p, v, why in dropped[:5]:
        print(f"   {p} = {v}  ({why})")

    out_yaml = f"/Users/siddharth/Study/3dv_project/verify_dump/demo_design_{gid}.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"design": design}, f, sort_keys=False, default_flow_style=False)
    print(f"[done] wrote {out_yaml}")

    # Simplified flat json for quick eyeballing: only the values actually kept.
    simple = {}
    for top_prefix, key, val in resolved:
        if is_split and top_prefix is not None and owning_container(top_prefix, key) != top_prefix:
            continue
        simple[key] = val
    for field, cands in meta_candidates.items():
        parts = field.split(".")
        node = design
        for p in parts[:-1]:
            node = node[p]
        simple[field] = node[parts[-1]]["v"]
    out_json = f"/Users/siddharth/Study/3dv_project/verify_dump/demo_design_{gid}_flat.json"
    with open(out_json, "w") as f:
        json.dump(simple, f, indent=1)
    print(f"[done] wrote {out_json}")

    print()
    print("Reference images for this garment (to compare against the GarmentCode render):")
    frames = images[gid]["frames"]
    frame0 = sorted(frames.keys(), key=int)[0]
    for p in frames[frame0]:
        print("  ", p)


if __name__ == "__main__":
    main()
