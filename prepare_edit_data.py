#!/usr/bin/env python3
"""prepare_edit_data.py

Build instruction-conditioned *editing* examples for the two float-readout
variants (``single_token`` / variant A and ``per_token`` / variant B), sharing
one canonical slot layout.

Source data
-----------
ChatGarment's editing split lives at
``data/chatgarment_data/training/editing/random_newgarments_textsewing.json``
(10k rows) and the held-out benchmark at
``data/chatgarment_data/evaluations/garment_edit_eva.json`` (134 rows). Each row:

  * ``conversations`` -- a ``human`` turn ("Adjust the old sewing pattern ...")
    carrying the *source* garment JSON with **real floats** plus an instruction
    dict, and a ``gpt`` turn: the *target* garment JSON with ``[SEG]`` tokens
    marking every float the model must produce,
  * ``all_floats`` -- the target float values, in the same **document order** as
    the ``[SEG]`` tokens.

Canonical slot layout (single source of truth)
----------------------------------------------
The image->params model froze a 152-slot layout in ``prepared_v2/schema.json``
(``cont_slots``), but those key-paths are namespaced under
``upperbody_garment.`` / ``lowerbody_garment.`` / ``wholebody_garment.``. The
editing JSON is a single flat garment, so its paths are *bare* (``collar.width``,
``pants.length``, ...). Stripping the body prefix from the 152 image slots yields
exactly **76 distinct bare paths**, and those 76 cover the editing split's float
paths 100%. So the editing canonical ordering is ``sorted()`` of those 76 bare
paths -- *derived from the existing frozen schema*, never hand-written. Both
variants share it. ``N_SLOTS = 76``.

Outputs (to ``--out``)
----------------------
  * ``schema_edit.json`` -- ``slots`` (the 76 ordered paths), ``slot_index``,
    per-slot ``ranges`` (observed [min, max] over train targets, for
    inference-time denorm/clamp), ``n_slots``, and the derivation provenance.
  * ``edit_train.jsonl`` / ``edit_val.jsonl`` / ``edit_eval.jsonl`` -- one JSON
    object per example (fields documented in ``build_example``).

Reuses ``parse_config`` / ``walk_config`` / ``is_owned_path`` / ``flatten_floats``
from ``prepare_data.py`` -- the same parsing the image pipeline uses.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from prepare_data import (
    parse_config,
    walk_config,
    is_owned_path,
    flatten_floats,
    _SEG_SENTINEL,
)

# --------------------------------------------------------------------------- #
# Special tokens (kept in sync with edit_model.py)
# --------------------------------------------------------------------------- #
ALLNUM_TOKEN = "<ALLNUM>"   # variant A: one sentinel closes the numeric section
VAL_TOKEN = "<VAL>"         # variant B: one per active float
PLACEHOLDER_A = "0"         # variant A: what an active float renders as in-body

_BODY_PREFIX_RE = re.compile(r"^(upperbody|lowerbody|wholebody)_garment\.")

# Markers in the human turn.
_M_SRC = "The old garment sewing pattern is:"
_M_INS = "And the text descriptions are:"


# --------------------------------------------------------------------------- #
# Canonical slot layout, derived from the frozen image schema
# --------------------------------------------------------------------------- #
def canonical_slots(image_schema_path):
    """Return ``(slots, slot_index)`` for the editing task.

    ``slots`` is the sorted list of bare (un-prefixed) float key-paths taken
    from the image model's frozen ``cont_slots``; ``slot_index`` maps path->idx.
    This is the single canonical ordering both variants use.
    """
    with open(image_schema_path) as f:
        sch = json.load(f)
    bare = sorted({_BODY_PREFIX_RE.sub("", p) for p in sch["cont_slots"]})
    slot_index = {p: i for i, p in enumerate(bare)}
    return bare, slot_index


# --------------------------------------------------------------------------- #
# Human-turn splitting
# --------------------------------------------------------------------------- #
def _extract_braced(s, start):
    """Return ``(substring, end)`` for the brace-balanced ``{...}`` at/after
    ``start``, respecting single-quoted strings (garment values never contain
    unquoted braces)."""
    i = s.index("{", start)
    depth = 0
    in_str = False
    j = i
    while j < len(s):
        c = s[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == "'":
                in_str = False
        else:
            if c == "'":
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def split_human(human_value):
    """Return ``(source_json_str, instruction_str)`` from a human turn."""
    src_str, end = _extract_braced(human_value, human_value.index(_M_SRC))
    ins_str, _ = _extract_braced(human_value, human_value.index(_M_INS, end))
    return src_str, ins_str


# --------------------------------------------------------------------------- #
# Canonical serialisation
# --------------------------------------------------------------------------- #
def _fmt_scalar(v):
    """Render a non-sentinel scalar in ChatGarment repr style."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Compact, order-independent numeric text (input side reads real floats).
        return repr(round(v, 5))
    # string (style name, etc.)
    return "'%s'" % v


def render(node, path, mode, seg_value_of, slot_index, active_out):
    """Serialise ``node`` with **recursively sorted dict keys** (canonical order).

    ``mode``:
      * ``"real"`` -- source side: real numbers rendered as-is (no sentinels).
      * ``"A"``    -- target variant A: each active float -> ``PLACEHOLDER_A``.
      * ``"B"``    -- target variant B: each active float -> ``VAL_TOKEN``.

    For target modes, every sentinel leaf appends ``(path, slot, value)`` to
    ``active_out`` **in emission order** -- this is the order variant B's
    ``<VAL>`` tokens appear in the text, so the training target values must be
    concatenated in exactly this order.
    """
    if isinstance(node, dict):
        parts = []
        for k in sorted(node.keys()):
            child = render(node[k], path + [str(k)], mode, seg_value_of,
                           slot_index, active_out)
            parts.append("'%s': %s" % (k, child))
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, list):
        parts = [render(v, path + [str(i)], mode, seg_value_of, slot_index,
                        active_out)
                 for i, v in enumerate(node)]
        return "[" + ", ".join(parts) + "]"
    if node == _SEG_SENTINEL:
        p = ".".join(path)
        active_out.append((p, slot_index.get(p), seg_value_of.get(p)))
        return PLACEHOLDER_A if mode == "A" else VAL_TOKEN
    return _fmt_scalar(node)


def collect_numeric_at_slots(node, path, slot_index, out):
    """Record numeric leaves whose path is a known slot (used for the *source*
    float vector, which carries real numbers rather than sentinels)."""
    if isinstance(node, dict):
        for k, v in node.items():
            collect_numeric_at_slots(v, path + [str(k)], slot_index, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            collect_numeric_at_slots(v, path + [str(i)], slot_index, out)
    elif isinstance(node, bool) or node is None or node == _SEG_SENTINEL:
        return
    elif isinstance(node, (int, float)):
        p = ".".join(path)
        if p in slot_index:
            out[p] = float(node)


# --------------------------------------------------------------------------- #
# Per-record example builder
# --------------------------------------------------------------------------- #
def build_example(rec, slots, slot_index):
    """Turn one raw editing record into the fields both variants consume.

    Returns a dict, or ``None`` if the record is malformed / out-of-schema.
    """
    n = len(slots)
    convs = {c["from"]: c["value"] for c in rec["conversations"]}
    human, gpt = convs.get("human"), convs.get("gpt")
    if human is None or gpt is None:
        return None

    # ---- source (real floats) + instruction ----
    try:
        src_str, ins_str = split_human(human)
        src_cfg = parse_config(src_str)
        tgt_cfg = parse_config(gpt)
    except Exception:
        return None

    # ---- target: document-order [SEG] paths aligned to all_floats ----
    seg_paths, cat_items, const_items = [], [], []
    walk_config(tgt_cfg, [], seg_paths, cat_items, const_items)
    floats = flatten_floats(rec.get("all_floats", []))
    if len(seg_paths) != len(floats):
        return None
    seg_value_of = {p: float(v) for p, v in zip(seg_paths, floats)
                    if is_owned_path(p)}
    # Every owned target float path must live in the canonical schema.
    if any(p not in slot_index for p in seg_value_of):
        return None

    # ---- canonical serialisations (single sorted-key walk each) ----
    active_A, active_B = [], []
    text_A = render(tgt_cfg, [], "A", seg_value_of, slot_index, active_A)
    text_B = render(tgt_cfg, [], "B", seg_value_of, slot_index, active_B)
    # Same walk order in both modes.
    assert [a[0] for a in active_A] == [b[0] for b in active_B]
    target_text_single = text_A + " " + ALLNUM_TOKEN
    target_text_per = text_B

    # ---- fixed-length target vector + active mask (canonical positions) ----
    target_floats = np.zeros(n, dtype=np.float32)
    active_mask = np.zeros(n, dtype=np.float32)
    for _p, slot, val in active_A:
        target_floats[slot] = val
        active_mask[slot] = 1.0

    # ---- per-<VAL> streams (emission order) for variant B ----
    active_slots = [slot for _p, slot, _v in active_B]     # slot idx per <VAL>
    active_vals = [val for _p, _s, val in active_B]        # target per <VAL>

    # Alignment invariants (the recurring silent-misalignment bug).
    assert len(active_slots) == int(active_mask.sum()), \
        "per-example <VAL> count must equal active_mask.sum()"
    assert text_B.count(VAL_TOKEN) == len(active_slots)
    assert text_A.count(" %s" % ALLNUM_TOKEN) == 0  # placeholder isn't a token

    # ---- source float vector (for changed-vs-preserved eval stratification) ----
    src_num = {}
    collect_numeric_at_slots(src_cfg, [], slot_index, src_num)
    source_floats = np.zeros(n, dtype=np.float32)
    source_mask = np.zeros(n, dtype=np.float32)
    for p, v in src_num.items():
        source_floats[slot_index[p]] = v
        source_mask[slot_index[p]] = 1.0

    # Canonical source serialisation (real floats on the input side).
    source_json_text = render(src_cfg, [], "real", {}, slot_index, [])

    return {
        "id": rec.get("id"),
        "target_name": rec.get("target_name"),
        "source_json_text": source_json_text,
        "instruction_text": ins_str,
        "target_text_single": target_text_single,   # variant A
        "target_text_per": target_text_per,          # variant B
        "active_slots": active_slots,                 # <VAL> emission order
        "active_vals": active_vals,                   # aligned to active_slots
        "target_floats": target_floats.tolist(),      # length N_SLOTS
        "active_mask": active_mask.tolist(),          # length N_SLOTS
        "source_floats": source_floats.tolist(),
        "source_mask": source_mask.tolist(),
    }


# --------------------------------------------------------------------------- #
# Prompt template (matches the ChatGarment editing phrasing in the data)
# --------------------------------------------------------------------------- #
def build_prompt(source_json_text, instruction_text):
    return (
        "Adjust the old sewing pattern according to the text descriptions.\n"
        "The old garment sewing pattern is: \n%s. \n"
        "And the text descriptions are: \n%s."
        % (source_json_text, instruction_text)
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_file(path, slots, slot_index, limit=None):
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    out, n_skip = [], 0
    for rec in data:
        ex = build_example(rec, slots, slot_index)
        if ex is None:
            n_skip += 1
            continue
        ex["prompt_text"] = build_prompt(ex["source_json_text"],
                                         ex["instruction_text"])
        out.append(ex)
    return out, n_skip


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-json",
                    default="data/chatgarment_data/training/editing/"
                            "random_newgarments_textsewing.json")
    ap.add_argument("--eval-json",
                    default="data/chatgarment_data/evaluations/garment_edit_eva.json")
    ap.add_argument("--image-schema", default="prepared_v2/schema.json",
                    help="frozen image schema the canonical ordering derives from")
    ap.add_argument("--out", default="prepared_edit")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap records per file (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    slots, slot_index = canonical_slots(args.image_schema)
    print(f"[schema] N_SLOTS={len(slots)} (derived from {args.image_schema})",
          flush=True)

    train_all, sk_tr = process_file(args.train_json, slots, slot_index, args.limit)
    eval_rows, sk_ev = process_file(args.eval_json, slots, slot_index, args.limit)
    print(f"[parse] train usable={len(train_all)} skipped={sk_tr}; "
          f"eval usable={len(eval_rows)} skipped={sk_ev}", flush=True)

    # Deterministic train/val split.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(train_all))
    rng.shuffle(idx)
    n_val = int(round(len(train_all) * args.val_frac))
    val_ids = set(idx[:n_val].tolist())
    train_rows = [r for i, r in enumerate(train_all) if i not in val_ids]
    val_rows = [r for i, r in enumerate(train_all) if i in val_ids]

    # Per-slot observed ranges over TRAIN active targets (inference denorm/clamp).
    tf = np.array([r["target_floats"] for r in train_rows], dtype=np.float32)
    am = np.array([r["active_mask"] for r in train_rows], dtype=np.float32)
    ranges = []
    for s in range(len(slots)):
        vals = tf[am[:, s] > 0, s]
        if vals.size:
            ranges.append([float(vals.min()), float(vals.max())])
        else:
            ranges.append([0.0, 1.0])

    with open(os.path.join(args.out, "schema_edit.json"), "w") as f:
        json.dump({
            "slots": slots,
            "slot_index": slot_index,
            "ranges": ranges,
            "n_slots": len(slots),
            "special_tokens": {"allnum": ALLNUM_TOKEN, "val": VAL_TOKEN,
                               "placeholder_a": PLACEHOLDER_A},
            "provenance": {
                "derived_from": args.image_schema,
                "method": "sorted(unique(strip_body_prefix(cont_slots)))",
            },
        }, f, indent=1)

    write_jsonl(os.path.join(args.out, "edit_train.jsonl"), train_rows)
    write_jsonl(os.path.join(args.out, "edit_val.jsonl"), val_rows)
    write_jsonl(os.path.join(args.out, "edit_eval.jsonl"), eval_rows)

    # Report edit locality stats over train (sanity + framing for eval later).
    changed = preserved = 0
    for r in train_rows:
        sf = np.array(r["source_floats"]); sm = np.array(r["source_mask"])
        tfv = np.array(r["target_floats"]); amv = np.array(r["active_mask"])
        both = (sm > 0) & (amv > 0)
        d = np.abs(sf - tfv) > 1e-4
        changed += int((both & d).sum())
        preserved += int((both & ~d).sum())
    print(f"[split] train={len(train_rows)} val={len(val_rows)} "
          f"eval={len(eval_rows)}", flush=True)
    print(f"[locality] active target slots also-in-source: changed={changed} "
          f"preserved={preserved}", flush=True)
    print(f"[done] wrote schema_edit.json + edit_{{train,val,eval}}.jsonl to "
          f"{args.out}", flush=True)


if __name__ == "__main__":
    main()
