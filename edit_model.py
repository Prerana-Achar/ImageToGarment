#!/usr/bin/env python3
"""edit_model.py

Instruction-conditioned garment editor with a *regressed* float readout. A
QLoRA-finetuned decoder LLM generates the target garment JSON as text, but the
continuous float parameters are **not** emitted as digits -- they are regressed
from the LLM's hidden states by a small MLP. Two readout variants share the
whole backbone/loss and are selected by ``variant``:

  * ``single_token`` (variant A, ChatGarment's scheme): the body renders every
    active float as a placeholder and a single ``<ALLNUM>`` sentinel closes the
    numeric section. Its hidden state feeds one MLP that outputs the whole
    length-``N_SLOTS`` float vector at once (positional).

  * ``per_token`` (variant B, ours): each active float renders as one ``<VAL>``
    token (preceded by its key). Each ``<VAL>`` hidden state feeds a **shared**
    MLP that outputs one scalar. The number of ``<VAL>`` tokens varies per
    example.

Non-negotiables honoured here:
  * regression heads are **plain linear** (no sigmoid/tanh); clamp only at
    inference (done in the inference script, not here);
  * the new special-token embedding rows are **trainable** (ChatGarment trains
    the token embeddings) -- via peft's ``trainable_token_indices``, which
    learns a delta for just those rows instead of the whole embedding matrix;
  * hidden states are indexed **at** the special token's own position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)

# Kept in sync with prepare_edit_data.py.
ALLNUM_TOKEN = "<ALLNUM>"
VAL_TOKEN = "<VAL>"
SPECIAL_TOKENS = [ALLNUM_TOKEN, VAL_TOKEN]

VARIANTS = ("single_token", "per_token")

# Default LoRA target modules for Qwen2 / Llama-style decoders.
_DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]


class RegressionHead(nn.Module):
    """Two-layer MLP, plain linear output (no squashing nonlinearity).

    Conditioning matters here: the LLM's last-layer hidden states have a large
    and backbone-dependent magnitude, so a default-initialised MLP starts by
    predicting values far outside the targets' [0, 1] range (observed initial
    MAE ~4.7 on a 0.5B backbone, ~10.5 on 3B). A LayerNorm on the input plus a
    small-weight / 0.5-bias final layer makes the head start at mid-range and
    converge from there. Crucially this is *not* a squashing nonlinearity -- the
    output stays linear and unbounded, so genuine 0.0/1.0 boundary slots keep
    full gradient. Both variants share this head class, so the A/B comparison
    is unaffected.
    """

    def __init__(self, d_model, hidden, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.constant_(self.fc2.bias, 0.5)   # mid-range start, not a clamp

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class GarmentEditModel(nn.Module):
    """Backbone + QLoRA + special tokens + the two regression heads."""

    def __init__(
        self,
        backbone_name="Qwen/Qwen2.5-3B-Instruct",
        variant="per_token",
        n_slots=76,
        head_hidden=512,
        lambda_num=0.1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_targets=None,
        load_4bit=True,
        grad_checkpointing=True,
        compute_dtype=torch.bfloat16,
        device_map=None,
        adapter_dir=None,
        tokenizer_dir=None,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.n_slots = n_slots
        self.lambda_num = lambda_num

        # ---- tokenizer + special tokens ----
        # When restoring a checkpoint, the saved tokenizer already contains the
        # special tokens; loading it back reproduces the exact vocab the trained
        # embeddings expect (add_special_tokens is then a no-op, so the ids and
        # the embedding size stay identical to training).
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        already = [t for t in SPECIAL_TOKENS
                   if t in self.tokenizer.get_vocab()]
        self._vocab_before = len(self.tokenizer) - len(already)
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        self.n_added = n_added
        self.allnum_id = self.tokenizer.convert_tokens_to_ids(ALLNUM_TOKEN)
        self.val_id = self.tokenizer.convert_tokens_to_ids(VAL_TOKEN)

        # ---- backbone (optionally 4-bit QLoRA) ----
        model_kwargs = dict(dtype=compute_dtype)
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        base = AutoModelForCausalLM.from_pretrained(backbone_name, **model_kwargs)
        # Resize BEFORE peft so the new rows exist in the module we save/train.
        base.resize_token_embeddings(len(self.tokenizer))
        self.d_model = base.config.hidden_size

        if load_4bit:
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=grad_checkpointing
            )
        elif grad_checkpointing:
            base.gradient_checkpointing_enable()

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets or _DEFAULT_LORA_TARGETS,
            # The new special tokens' embeddings must be trainable (ChatGarment
            # trains them), but `modules_to_save=["embed_tokens","lm_head"]`
            # would make the *entire* 151936xD matrix trainable -- ~311M params
            # per copy, whose Adam states alone OOM a 16GB A4000. peft's
            # trainable_token_indices learns a delta for just these rows
            # (n_added x D params) and follows the tied lm_head automatically.
            trainable_token_indices={"embed_tokens": [self.allnum_id,
                                                     self.val_id]},
        )
        if adapter_dir is not None:
            self.backbone = PeftModel.from_pretrained(base, adapter_dir,
                                                      is_trainable=False)
        else:
            self.backbone = get_peft_model(base, lora)
            if grad_checkpointing:
                # output_hidden_states must stay differentiable under ckpt.
                self.backbone.enable_input_require_grads()

        # ---- regression heads (fp32 for stable regression) ----
        self.headA = RegressionHead(self.d_model, head_hidden, n_slots).float()
        self.headB = RegressionHead(self.d_model, head_hidden, 1).float()

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        params += list(self.headA.parameters())
        params += list(self.headB.parameters())
        return params

    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, labels,
                target_floats=None, active_mask=None, target_vals_flat=None):
        """Compute CE (target JSON tokens) + lambda_num * L1 (regressed floats).

        Shapes:
          input_ids / attention_mask / labels : (B, T)
          target_floats / active_mask         : (B, N_SLOTS)   [variant A]
          target_vals_flat                    : (num_val_total,) [variant B]
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        ce = out.loss
        h = out.hidden_states[-1].float()          # (B, T, D)
        B = input_ids.shape[0]

        if self.variant == "single_token":
            is_sent = (input_ids == self.allnum_id)
            # exactly one <ALLNUM> per sequence
            counts = is_sent.sum(dim=1)
            assert torch.all(counts == 1), \
                f"variant A needs exactly one <ALLNUM>/seq, got {counts.tolist()}"
            pos = is_sent.float().argmax(dim=1)     # (B,)
            h_num = h[torch.arange(B, device=h.device), pos]   # (B, D)
            pred = self.headA(h_num)                # (B, N_SLOTS)
            m = active_mask.float()
            denom = m.sum().clamp_min(1.0)
            num_l1 = (F.l1_loss(pred, target_floats, reduction="none")
                      * m).sum() / denom
            with torch.no_grad():
                mae = num_l1.detach()
            n_active = int(m.sum().item())
        else:  # per_token
            mask = (input_ids == self.val_id)       # (B, T)
            h_val = h[mask]                          # (num_val_total, D) row-major
            assert h_val.shape[0] == target_vals_flat.shape[0], (
                f"<VAL> hidden count {h_val.shape[0]} != target count "
                f"{target_vals_flat.shape[0]} -- alignment broken")
            pred = self.headB(h_val).squeeze(-1)     # (num_val_total,)
            num_l1 = F.l1_loss(pred, target_vals_flat)
            mae = num_l1.detach()
            n_active = int(h_val.shape[0])

        loss = ce + self.lambda_num * num_l1
        return {
            "loss": loss,
            "ce": ce.detach(),
            "num_l1": num_l1.detach(),
            "mae": mae,
            "n_active": n_active,
        }

    # ------------------------------------------------------------------ #
    def save_pretrained(self, out_dir):
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.backbone.save_pretrained(out_dir)          # LoRA + saved modules
        self.tokenizer.save_pretrained(out_dir)
        torch.save(
            {
                "headA": self.headA.state_dict(),
                "headB": self.headB.state_dict(),
                "variant": self.variant,
                "n_slots": self.n_slots,
                "d_model": self.d_model,
                "lambda_num": self.lambda_num,
                "vocab_before": self._vocab_before,
            },
            os.path.join(out_dir, "heads.pt"),
        )

    def load_heads(self, out_dir, map_location="cpu"):
        import os
        ckpt = torch.load(os.path.join(out_dir, "heads.pt"),
                          map_location=map_location)
        self.headA.load_state_dict(ckpt["headA"])
        self.headB.load_state_dict(ckpt["headB"])
        return ckpt

    @classmethod
    def from_checkpoint(cls, ckpt_dir, backbone_name, load_4bit=True,
                        device_map=None, compute_dtype=torch.bfloat16):
        """Rebuild a trained editor for inference: base backbone + saved LoRA /
        embeddings (via ``adapter_dir``) + the regression heads."""
        import os
        meta = torch.load(os.path.join(ckpt_dir, "heads.pt"),
                          map_location="cpu")
        model = cls(
            backbone_name=backbone_name,
            variant=meta["variant"],
            n_slots=meta["n_slots"],
            lambda_num=meta.get("lambda_num", 0.1),
            load_4bit=load_4bit,
            grad_checkpointing=False,
            device_map=device_map,
            compute_dtype=compute_dtype,
            adapter_dir=ckpt_dir,
            tokenizer_dir=ckpt_dir,
        )
        model.headA.load_state_dict(meta["headA"])
        model.headB.load_state_dict(meta["headB"])
        model.eval()
        return model


def build_model(**kwargs):
    """Convenience constructor."""
    return GarmentEditModel(**kwargs)
