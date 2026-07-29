#!/usr/bin/env python3
"""infer_edit.py

Two-pass inference for the garment editor (avoids hooking into ``generate``):

  1. Greedy-decode the full target token sequence from the prompt.
  2. Run **one** forward pass over ``[prompt + generated]`` with
     ``output_hidden_states=True``.
  3. Index the special-token positions exactly as in training and run the
     head(s) to get the floats.
  4. Clamp each slot to its observed range and splice the numbers back into the
     parsed JSON (replacing placeholders / ``<VAL>`` in document order).

Also supports ``--teacher-forced``: instead of generating, feed the reference
target text and read the heads at its special tokens. This isolates the float
*readout* quality (the A/B question) from JSON-generation quality, so both
variants are scored on the identical ground-truth structure.

Writes one JSON object per example to ``--out`` (pred JSON, pred float vector,
active mask, and bookkeeping), consumed by ``eval_edit.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from edit_model import GarmentEditModel, ALLNUM_TOKEN, VAL_TOKEN
from prepare_data import parse_config, _SEG_SENTINEL

_VAL_SENTINEL = "__EDIT_VAL__"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def clean_generated(text):
    """Strip trailing special/eos markup and the variant-A sentinel from a
    decoded target so it parses as a config dict."""
    for tok in (ALLNUM_TOKEN, "<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, " ")
    return text.strip()


def parse_target_text(text, variant):
    """Parse a (generated or reference) target into a Python dict.

    Variant B substitutes ``<VAL>`` -> a quoted sentinel so ``ast.literal_eval``
    accepts it; the sentinels are replaced with real numbers during reassembly.
    """
    text = clean_generated(text)
    if variant == "per_token":
        text = text.replace(VAL_TOKEN, "'%s'" % _VAL_SENTINEL)
    return parse_config(text)


def doc_order_slot_leaves(node, path, slot_index, out):
    """Append ``(dotted_path, is_sentinel)`` for every leaf whose path is a
    known float slot, in document order (matches ``<VAL>`` token order)."""
    if isinstance(node, dict):
        for k, v in node.items():
            doc_order_slot_leaves(v, path + [str(k)], slot_index, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            doc_order_slot_leaves(v, path + [str(i)], slot_index, out)
    else:
        p = ".".join(path)
        if p in slot_index:
            out.append((p, node == _VAL_SENTINEL))


def set_at_path(root, dotted, value):
    parts = dotted.split(".")
    node = root
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def denorm_clamp(value, lo, hi):
    """Clamp to the slot's observed [min, max] (the data's floats are already in
    GarmentCode's normalised space, so 'denorm' here is just a range clamp)."""
    return float(min(max(value, lo), hi))


# --------------------------------------------------------------------------- #
# Core: read heads off a token sequence
# --------------------------------------------------------------------------- #
@torch.no_grad()
def read_heads(model, full_ids, device):
    """Forward ``full_ids`` (1, T) and return last-layer hidden states (T, D)."""
    attn = torch.ones_like(full_ids)
    out = model.backbone(input_ids=full_ids, attention_mask=attn,
                         output_hidden_states=True)
    return out.hidden_states[-1].float()[0]        # (T, D)


def scatter_predictions(model, schema, target_dict, full_ids, target_text):
    """Given a parsed target dict + the token ids it was tokenised from, run the
    head(s) and return ``(pred_floats[N], active_mask[N], filled_dict)``."""
    slot_index = schema["slot_index"]
    ranges = schema["ranges"]
    n = schema["n_slots"]
    ids = full_ids[0]
    h = read_heads(model, full_ids, full_ids.device)

    pred_floats = np.zeros(n, dtype=np.float32)
    active_mask = np.zeros(n, dtype=np.float32)
    filled = copy.deepcopy(target_dict)

    # Document-order slot leaves (same order as special tokens in the text).
    leaves = []
    doc_order_slot_leaves(target_dict, [], slot_index, leaves)

    if model.variant == "single_token":
        pos_all = (ids == model.allnum_id).nonzero(as_tuple=True)[0]
        if len(pos_all) == 0:
            return pred_floats, active_mask, filled, False
        h_num = h[pos_all[0]]
        pred = model.headA(h_num).detach().cpu().numpy()     # (N,)
        for path, _is_sent in leaves:
            s = slot_index[path]
            lo, hi = ranges[s]
            v = denorm_clamp(float(pred[s]), lo, hi)
            pred_floats[s] = v
            active_mask[s] = 1.0
            set_at_path(filled, path, round(v, 5))
        ok = True
    else:  # per_token
        val_pos = (ids == model.val_id).nonzero(as_tuple=True)[0]
        if len(val_pos):
            h_val = h[val_pos]                                # (k, D)
            preds = model.headB(h_val).squeeze(-1).detach().cpu().numpy()
        else:
            preds = np.zeros(0, dtype=np.float32)
        # Assign preds to sentinel leaves in document order (== <VAL> order).
        sent_leaves = [p for p, is_s in leaves if is_s]
        k = min(len(sent_leaves), len(preds))
        for i in range(k):
            path = sent_leaves[i]
            s = slot_index[path]
            lo, hi = ranges[s]
            v = denorm_clamp(float(preds[i]), lo, hi)
            pred_floats[s] = v
            active_mask[s] = 1.0
            set_at_path(filled, path, round(v, 5))
        ok = (len(sent_leaves) == len(preds))
    return pred_floats, active_mask, filled, ok


# --------------------------------------------------------------------------- #
# Prompt / generation
# --------------------------------------------------------------------------- #
def prompt_ids(model, prompt_text, device):
    enc = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)


@torch.no_grad()
def generate_target(model, p_ids, max_new_tokens=768):
    gen = model.backbone.generate(
        input_ids=p_ids,
        attention_mask=torch.ones_like(p_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    return gen                                    # (1, prompt+gen)


def predict_one(model, schema, example, device, teacher_forced=False,
                max_new_tokens=768):
    """Return a prediction dict for one example."""
    p_ids = prompt_ids(model, example["prompt_text"], device)
    variant = model.variant

    if teacher_forced:
        tgt_text = (example["target_text_single"] if variant == "single_token"
                    else example["target_text_per"])
        tgt_ids = model.tokenizer(tgt_text, add_special_tokens=False,
                                  return_tensors="pt")["input_ids"].to(device)
        full_ids = torch.cat([p_ids, tgt_ids], dim=1)
        gen_text = tgt_text
    else:
        full = generate_target(model, p_ids, max_new_tokens)
        gen_ids = full[0, p_ids.shape[1]:]
        gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=False)
        full_ids = full

    try:
        tgt_dict = parse_target_text(gen_text, variant)
        parse_ok = True
    except Exception:
        tgt_dict = None
        parse_ok = False

    result = {
        "id": example.get("id"),
        "target_name": example.get("target_name"),
        "teacher_forced": teacher_forced,
        "parse_ok": parse_ok,
        "gen_text": gen_text,
    }
    if not parse_ok:
        result["pred_floats"] = [0.0] * schema["n_slots"]
        result["active_mask"] = [0.0] * schema["n_slots"]
        result["align_ok"] = False
        result["pred_json"] = None
        return result

    pred_floats, active_mask, filled, align_ok = scatter_predictions(
        model, schema, tgt_dict, full_ids, gen_text)
    result["pred_floats"] = pred_floats.tolist()
    result["active_mask"] = active_mask.tolist()
    result["align_ok"] = bool(align_ok)
    result["pred_json"] = filled
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (final/step_*)")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="prepared_edit/edit_eval.jsonl")
    ap.add_argument("--schema", default="prepared_edit/schema_edit.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-forced", action="store_true",
                    help="read heads off the reference target (isolates readout)")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_4bit = (not args.no_4bit) and device == "cuda"
    schema = json.load(open(args.schema))

    print(f"[load] {args.ckpt} (backbone={args.backbone} 4bit={load_4bit})",
          flush=True)
    model = GarmentEditModel.from_checkpoint(
        args.ckpt, args.backbone, load_4bit=load_4bit,
        device_map={"": 0} if load_4bit else None)
    if not load_4bit:
        model.backbone.to(device)
    model.headA.to(device); model.headB.to(device)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]

    n_parse = n_align = 0
    with open(args.out, "w") as f:
        for i, ex in enumerate(rows):
            r = predict_one(model, schema, ex, device,
                            teacher_forced=args.teacher_forced,
                            max_new_tokens=args.max_new_tokens)
            n_parse += int(r["parse_ok"]); n_align += int(r["align_ok"])
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 20 == 0:
                print(f"[infer] {i+1}/{len(rows)} parse_ok={n_parse} "
                      f"align_ok={n_align}", flush=True)

    print(f"[done] {len(rows)} examples -> {args.out} "
          f"(parse_ok={n_parse}, align_ok={n_align})", flush=True)


if __name__ == "__main__":
    main()
