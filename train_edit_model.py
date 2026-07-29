#!/usr/bin/env python3
"""train_edit_model.py

Single training loop for both float-readout variants of the garment editor.
``--variant {single_token, per_token}`` selects the head + which special token
the collator's targets align to; everything else (data, backbone, QLoRA, losses)
is identical, so the two runs are directly comparable.

Data comes from ``prepare_edit_data.py`` (``prepared_edit/``):
  * ``schema_edit.json`` -- canonical slot layout (N_SLOTS).
  * ``edit_train.jsonl`` / ``edit_val.jsonl`` -- examples carrying both target
    serialisations, ``target_floats`` + ``active_mask`` (variant A) and
    ``active_vals`` in ``<VAL>`` emission order (variant B).

Losses: ``CE(target JSON tokens) + lambda_num * L1(regressed floats)`` (prompt
tokens masked to -100). CE, L1 and active-slot MAE are logged separately.

Example:
    python train_edit_model.py --variant per_token --out runs/edit_per_token
    python train_edit_model.py --variant single_token --out runs/edit_single
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from edit_model import GarmentEditModel


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EditJsonlDataset(Dataset):
    """Tokenise prompt + variant-specific target; mask the prompt in the labels.

    Each item returns python lists (padded by the collator):
      input_ids, labels        -- (T,)  prompt masked to -100 in labels
      target_floats, active_mask -- (N_SLOTS,)  [variant A]
      active_vals              -- (n_active,) in <VAL> emission order [variant B]
    """

    def __init__(self, jsonl_path, tokenizer, variant, max_len=1024, limit=None):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        if limit:
            self.rows = self.rows[:limit]
        self.tok = tokenizer
        self.variant = variant
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id
        self.n_trunc = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        msgs = [{"role": "user", "content": r["prompt_text"]}]
        prompt_ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )["input_ids"]

        tgt_text = (r["target_text_single"] if self.variant == "single_token"
                    else r["target_text_per"])
        tgt_ids = self.tok(tgt_text, add_special_tokens=False)["input_ids"]
        tgt_ids = tgt_ids + [self.eos_id]

        input_ids = prompt_ids + tgt_ids
        labels = [-100] * len(prompt_ids) + tgt_ids

        if len(input_ids) > self.max_len:
            # Keep the target intact; drop earliest (masked) prompt tokens.
            cut = len(input_ids) - self.max_len
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            self.n_trunc += 1

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "target_floats": r["target_floats"],
            "active_mask": r["active_mask"],
            "active_vals": r["active_vals"],
        }
        return item


def make_collator(pad_id, variant):
    def collate(batch):
        T = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)

        out = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
        if variant == "single_token":
            out["target_floats"] = torch.tensor(
                np.array([b["target_floats"] for b in batch]), dtype=torch.float32)
            out["active_mask"] = torch.tensor(
                np.array([b["active_mask"] for b in batch]), dtype=torch.float32)
        else:
            # Concatenate per example, in batch order, each example's active_vals
            # in <VAL> emission order == the row-major order of h[input_ids==VAL].
            flat = []
            for b in batch:
                flat.extend(b["active_vals"])
            out["target_vals_flat"] = torch.tensor(flat, dtype=torch.float32)
        return out

    return collate


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, variant, max_batches=None):
    model.eval()
    agg = {"ce": 0.0, "num_l1": 0.0, "mae": 0.0, "n": 0}
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = _to_device(batch, device)
        r = model(**_forward_kwargs(batch, variant))
        agg["ce"] += float(r["ce"])
        agg["num_l1"] += float(r["num_l1"])
        agg["mae"] += float(r["mae"])
        agg["n"] += 1
    model.train()
    n = max(agg["n"], 1)
    return {k: agg[k] / n for k in ("ce", "num_l1", "mae")}


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _forward_kwargs(batch, variant):
    kw = dict(input_ids=batch["input_ids"],
              attention_mask=batch["attention_mask"],
              labels=batch["labels"])
    if variant == "single_token":
        kw["target_floats"] = batch["target_floats"]
        kw["active_mask"] = batch["active_mask"]
    else:
        kw["target_vals_flat"] = batch["target_vals_flat"]
    return kw


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    choices=["single_token", "per_token"])
    ap.add_argument("--data-dir", default="prepared_edit")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA + embeddings lr")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="regression head lr")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--lambda-num", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train examples (smoke test)")
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_4bit = (not args.no_4bit) and device == "cuda"
    if args.no_4bit is False and device == "cpu":
        print("[warn] no CUDA -> falling back to fp (bitsandbytes needs a GPU)",
              flush=True)

    schema = json.load(open(os.path.join(args.data_dir, "schema_edit.json")))
    n_slots = schema["n_slots"]

    print(f"[build] backbone={args.backbone} variant={args.variant} "
          f"4bit={load_4bit} device={device}", flush=True)
    model = GarmentEditModel(
        backbone_name=args.backbone,
        variant=args.variant,
        n_slots=n_slots,
        head_hidden=args.head_hidden,
        lambda_num=args.lambda_num,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_4bit=load_4bit,
        grad_checkpointing=not args.no_grad_ckpt,
        device_map={"": 0} if load_4bit else None,
    )
    if not load_4bit:
        model.backbone.to(device)
    model.headA.to(device)
    model.headB.to(device)
    print(f"[build] trainable params: {model.num_trainable()/1e6:.2f}M", flush=True)

    tok = model.tokenizer
    train_ds = EditJsonlDataset(os.path.join(args.data_dir, "edit_train.jsonl"),
                                tok, args.variant, args.max_len, args.limit)
    val_path = os.path.join(args.data_dir, "edit_val.jsonl")
    val_ds = (EditJsonlDataset(val_path, tok, args.variant, args.max_len,
                               args.val_limit) if os.path.exists(val_path) else None)
    collate = make_collator(tok.pad_token_id, args.variant)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers)
                  if val_ds else None)
    print(f"[data] train={len(train_ds)} val={len(val_ds) if val_ds else 0}",
          flush=True)

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = (args.max_steps if args.max_steps
                   else int(steps_per_epoch * args.epochs))
    warmup = max(int(total_steps * args.warmup_frac), 1)

    head_params = list(model.headA.parameters()) + list(model.headB.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.backbone.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_scale)

    hist_path = os.path.join(args.out, "history.jsonl")
    hist = open(hist_path, "a")
    model.train()
    optim.zero_grad()
    gstep = 0
    micro = 0
    run = {"loss": 0.0, "ce": 0.0, "num_l1": 0.0, "mae": 0.0, "k": 0}
    t0 = time.time()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for batch in train_loader:
            batch = _to_device(batch, device)
            r = model(**_forward_kwargs(batch, args.variant))
            loss = r["loss"] / args.grad_accum
            loss.backward()
            run["loss"] += float(r["loss"]); run["ce"] += float(r["ce"])
            run["num_l1"] += float(r["num_l1"]); run["mae"] += float(r["mae"])
            run["k"] += 1
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                               args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad()
                gstep += 1

                if gstep % args.log_every == 0:
                    k = run["k"]
                    msg = (f"[step {gstep}/{total_steps}] "
                           f"loss={run['loss']/k:.4f} ce={run['ce']/k:.4f} "
                           f"L1={run['num_l1']/k:.4f} MAE={run['mae']/k:.4f} "
                           f"lr={sched.get_last_lr()[0]:.2e} "
                           f"({(time.time()-t0)/gstep:.2f}s/step)")
                    print(msg, flush=True)
                    hist.write(json.dumps({
                        "step": gstep, "split": "train",
                        "loss": run["loss"]/k, "ce": run["ce"]/k,
                        "num_l1": run["num_l1"]/k, "mae": run["mae"]/k}) + "\n")
                    hist.flush()
                    run = {"loss": 0.0, "ce": 0.0, "num_l1": 0.0, "mae": 0.0, "k": 0}

                if val_loader and gstep % args.val_every == 0:
                    v = evaluate(model, val_loader, device, args.variant,
                                 max_batches=args.val_limit)
                    print(f"[val {gstep}] ce={v['ce']:.4f} L1={v['num_l1']:.4f} "
                          f"MAE={v['mae']:.4f}", flush=True)
                    hist.write(json.dumps({"step": gstep, "split": "val", **v})
                               + "\n"); hist.flush()

                if gstep % args.save_every == 0:
                    ckpt = os.path.join(args.out, f"step_{gstep}")
                    model.save_pretrained(ckpt)
                    print(f"[save] {ckpt}", flush=True)

                if gstep >= total_steps:
                    done = True
                    break
        if args.max_steps is None and epoch >= math.ceil(args.epochs) and not done:
            done = True

    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    if train_ds.n_trunc:
        print(f"[warn] {train_ds.n_trunc} examples truncated to max_len={args.max_len}",
              flush=True)
    print(f"[done] variant={args.variant} steps={gstep} -> {final}", flush=True)
    hist.close()


if __name__ == "__main__":
    main()
