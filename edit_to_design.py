#!/usr/bin/env python3
"""edit_to_design.py

Bridge the editor's output to the renderer. ``infer_edit.py`` emits ``pred_json``
-- the edited garment as a flat ChatGarment-style nested dict whose float slots
hold **normalised** values (the head's regressed outputs). GarmentCode's renderer
(``render_garmentcode.py``) instead needs a *design YAML*: the full component
tree where every leaf is a ``{v, range, type}`` spec and floats are in real
(denormalised) units.

This walks ``pred_json`` and writes each leaf onto a copy of a template design
tree (the superset ``verify_dump/demo_design_v2_1327.yaml`` carries every
component + ``meta`` selector), exactly as ``prediction_to_yaml.py`` does for the
image model:
  * float slots (paths in ``schema_edit.json``) are denormalised with the
    template spec's own range:  raw = lo + norm * (hi - lo);
  * everything else (constants, categoricals, ``meta.*``) is written verbatim,
    cast to the spec's type.

One ``--preds`` JSONL (from infer_edit) -> one design YAML per example, named by
id, under ``--out-dir``. Rows whose ``pred_json`` is null (generation failed to
parse) are skipped and reported.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import yaml

from prediction_to_yaml import get_spec, cast_value, numeric_range


def leaves(node, path, out):
    """Yield (dotted_path, value) for every scalar leaf of a nested dict/list."""
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    else:
        out.append((".".join(path), node))


def apply_pred_json(design, pred_json, slot_index):
    """Write one pred_json onto a design tree copy. Returns list of warnings."""
    warnings = []
    flat = []
    leaves(pred_json, [], flat)
    for path, val in flat:
        try:
            spec = get_spec(design, path)          # template leaf {v,range,type}
        except Exception as exc:
            warnings.append(f"{path}: no template spec ({exc})")
            continue
        try:
            if path in slot_index and isinstance(val, (int, float)):
                lo, hi = numeric_range(spec)
                raw = lo + float(val) * (hi - lo)
                spec["v"] = cast_value(raw, spec)
            else:
                spec["v"] = cast_value(val, spec)
        except Exception as exc:
            warnings.append(f"{path}: {exc}")
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="infer_edit.py output JSONL")
    ap.add_argument("--template", default="verify_dump/demo_design_v2_1327.yaml")
    ap.add_argument("--schema", default="prepared_edit/schema_edit.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.template) as f:
        template = yaml.safe_load(f)
    base_design = template["design"]
    slot_index = json.load(open(args.schema))["slot_index"]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.preds)]
    if args.limit:
        rows = rows[:args.limit]

    n_ok = n_skip = 0
    for r in rows:
        rid = r.get("id")
        pj = r.get("pred_json")
        if pj is None:
            n_skip += 1
            print(f"[skip] id={rid}: pred_json is null (generation parse failed)",
                  flush=True)
            continue
        design = copy.deepcopy(base_design)
        warns = apply_pred_json(design, pj, slot_index)
        out_path = os.path.join(args.out_dir, f"edit_{rid}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump({"design": design}, f, sort_keys=False,
                           default_flow_style=False)
        n_ok += 1
        msg = f"[ok] id={rid} -> {out_path}"
        if warns:
            msg += f"  ({len(warns)} unmapped paths)"
        print(msg, flush=True)

    print(f"[done] designs written={n_ok} skipped={n_skip} -> {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
