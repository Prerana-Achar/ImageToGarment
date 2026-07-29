#!/usr/bin/env python3
"""eval_edit.py

Score editor predictions and compare the two float-readout variants head to head.

Metrics (all computed against the reference target in ``edit_eval.jsonl``):
  * **Active-slot MAE** -- mean |pred - target| over slots active in the target.
  * **Changed-field MAE** vs **Preserved-field MAE** -- edit *locality*. Active
    target slots that also exist in the source are split by whether the source
    and target values differ (>eps). Changed = the edit should have moved them;
    Preserved = the edit should have left them alone. Reporting both is the
    whole point of editing.
  * **Discrete-field accuracy** -- exact-match rate over the non-float JSON
    leaves (styles, bools, null, integer constants). Only meaningful for
    *generated* predictions; under ``--teacher-forced`` the structure is copied
    from the reference so this is ~100% by construction.
  * **Parse rate / alignment rate** -- fraction of generated targets that parsed
    and whose per-token count matched.

Usage:
    python eval_edit.py --schema prepared_edit/schema_edit.json \\
        --ref prepared_edit/edit_eval.jsonl \\
        --preds single_token:preds_A.jsonl per_token:preds_B.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from prepare_data import parse_config
from infer_edit import clean_generated

_EPS = 1e-4


def discrete_leaves(node, path, slot_index, out):
    """Collect (path, value) for every leaf that is NOT a float slot."""
    if isinstance(node, dict):
        for k, v in node.items():
            discrete_leaves(v, path + [str(k)], slot_index, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            discrete_leaves(v, path + [str(i)], slot_index, out)
    else:
        p = ".".join(path)
        if p not in slot_index:
            out[p] = node


def get_at_path(root, dotted):
    node = root
    for p in dotted.split("."):
        try:
            node = node[int(p)] if isinstance(node, list) else node[p]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    return node


_MISSING = object()


def score(preds_path, ref_by_id, schema):
    slot_index = schema["slot_index"]
    rows = [json.loads(l) for l in open(preds_path)]

    # Float accumulators.
    ae_all, ae_changed, ae_preserved = [], [], []
    # Discrete accumulators.
    disc_correct = disc_total = 0
    n_parse = n_align = n = 0

    for r in rows:
        ref = ref_by_id.get(r.get("id"))
        if ref is None:
            continue
        n += 1
        n_parse += int(r.get("parse_ok", False))
        n_align += int(r.get("align_ok", False))

        pred = np.array(r["pred_floats"], dtype=np.float32)
        tgt = np.array(ref["target_floats"], dtype=np.float32)
        amask = np.array(ref["active_mask"], dtype=np.float32)
        src = np.array(ref["source_floats"], dtype=np.float32)
        smask = np.array(ref["source_mask"], dtype=np.float32)

        active = amask > 0
        ae = np.abs(pred - tgt)
        ae_all.extend(ae[active].tolist())

        # Locality split: active target slots that also live in the source.
        both = active & (smask > 0)
        moved = both & (np.abs(src - tgt) > _EPS)
        kept = both & (np.abs(src - tgt) <= _EPS)
        ae_changed.extend(ae[moved].tolist())
        ae_preserved.extend(ae[kept].tolist())

        # Discrete-field accuracy (generated preds only make this meaningful).
        if r.get("pred_json") is not None and ref.get("_target_dict") is not None:
            ref_disc = {}
            discrete_leaves(ref["_target_dict"], [], slot_index, ref_disc)
            for p, v in ref_disc.items():
                pv = get_at_path(r["pred_json"], p)
                disc_total += 1
                disc_correct += int(pv is not _MISSING and pv == v)

    def m(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "n": n,
        "parse_rate": n_parse / max(n, 1),
        "align_rate": n_align / max(n, 1),
        "mae_active": m(ae_all),
        "mae_changed": m(ae_changed),
        "mae_preserved": m(ae_preserved),
        "n_changed": len(ae_changed),
        "n_preserved": len(ae_preserved),
        "disc_acc": (disc_correct / disc_total) if disc_total else float("nan"),
        "n_disc": disc_total,
    }


def load_reference(ref_path, schema):
    """Index reference examples by id, attaching a parsed target dict (for
    discrete-field accuracy) built from the variant-A target text."""
    ref_by_id = {}
    for l in open(ref_path):
        ex = json.loads(l)
        try:
            ex["_target_dict"] = parse_config(
                clean_generated(ex["target_text_single"]))
        except Exception:
            ex["_target_dict"] = None
        ref_by_id[ex.get("id")] = ex
    return ref_by_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="prepared_edit/schema_edit.json")
    ap.add_argument("--ref", default="prepared_edit/edit_eval.jsonl")
    ap.add_argument("--preds", nargs="+", required=True,
                    metavar="LABEL:PATH",
                    help="one or more labelled prediction files, e.g. "
                         "single_token:preds_A.jsonl per_token:preds_B.jsonl")
    ap.add_argument("--out", default=None, help="optional metrics.json path")
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    ref_by_id = load_reference(args.ref, schema)

    results = {}
    for spec in args.preds:
        label, path = spec.split(":", 1)
        results[label] = score(path, ref_by_id, schema)

    # ---- comparison table ----
    cols = [
        ("mae_active", "MAE(active)", "{:.4f}"),
        ("mae_changed", "MAE(changed)", "{:.4f}"),
        ("mae_preserved", "MAE(preserved)", "{:.4f}"),
        ("disc_acc", "DiscAcc", "{:.3f}"),
        ("parse_rate", "Parse", "{:.3f}"),
        ("align_rate", "Align", "{:.3f}"),
        ("n", "N", "{:d}"),
    ]
    label_w = max(len(l) for l in results) if results else 8
    header = "variant".ljust(label_w) + "  " + "  ".join(
        h.rjust(14) for _, h, _ in cols)
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        cells = []
        for key, _h, fmt in cols:
            v = r[key]
            cells.append((fmt.format(v) if not (isinstance(v, float)
                          and np.isnan(v)) else "nan").rjust(14))
        print(label.ljust(label_w) + "  " + "  ".join(cells))

    if len(results) == 2:
        (la, ra), (lb, rb) = list(results.items())
        print(f"\n[A/B] {la} vs {lb}: "
              f"dMAE(active)={rb['mae_active']-ra['mae_active']:+.4f}  "
              f"dMAE(changed)={rb['mae_changed']-ra['mae_changed']:+.4f}  "
              f"dMAE(preserved)={rb['mae_preserved']-ra['mae_preserved']:+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
