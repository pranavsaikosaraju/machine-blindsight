"""Load a Hugging Face causal LM, read its yes/no answer, and cache activations.

Works for base models (plain "Question/Answer" prompt) and chat models (the
tokenizer's chat template). For every item we record:

* the model's *report*: which of the two answer words it prefers, read from the
  next-token logits at the answer position, and its confidence (softmax over the
  two answer words only);
* the residual-stream activation at that same final prompt position for every
  layer (hidden_states[0] = token embeddings, hidden_states[L] = after block L).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "Answer the question with a single word: {pos} or {neg}."


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    tok: AutoTokenizer
    is_chat: bool
    device: torch.device


def load_model(name: str, dtype=torch.float16, random_init: bool = False) -> LoadedModel:
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if random_init:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
    model.eval()
    lname = name.lower()
    is_chat = getattr(tok, "chat_template", None) is not None and (
        "instruct" in lname or "-it" in lname or "chat" in lname)
    device = next(model.parameters()).device
    return LoadedModel(name, model, tok, is_chat, device)


def answer_words(fmt: str) -> tuple[str, str]:
    return ("Yes", "No") if fmt == "yesno" else ("True", "False")


def build_prompt(lm: LoadedModel, question: str, fmt: str) -> str:
    pos, neg = answer_words(fmt)
    if lm.is_chat:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT.format(pos=pos, neg=neg)},
            {"role": "user", "content": question},
        ]
        try:
            return lm.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # some templates (e.g. Gemma) reject a system role
            msgs = [{"role": "user", "content": SYSTEM_PROMPT.format(pos=pos, neg=neg) + "\n\n" + question}]
            return lm.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT.format(pos=pos, neg=neg)}\n\nQuestion: {question}\nAnswer:"


def _first_token_ids(tok, word: str) -> list[int]:
    """Token ids that can start `word` as the next token (with and without a leading space)."""
    ids = set()
    for variant in (word, " " + word, word.lower(), " " + word.lower()):
        enc = tok.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def answer_token_ids(lm: LoadedModel, fmt: str) -> tuple[list[int], list[int]]:
    pos, neg = answer_words(fmt)
    p, n = _first_token_ids(lm.tok, pos), _first_token_ids(lm.tok, neg)
    overlap = set(p) & set(n)
    return [i for i in p if i not in overlap], [i for i in n if i not in overlap]


def _logsumexp(x: torch.Tensor, idx: list[int]) -> torch.Tensor:
    return torch.logsumexp(x[:, idx].float(), dim=-1)


@torch.no_grad()
def run_items(lm: LoadedModel, df, batch_size: int = 16, layers: list[int] | None = None,
              hook_fn=None, cache: bool = True, progress: bool = True):
    """Run every row of df. Returns (report_df_columns dict, activations array or None).

    activations: float32 array [n_items, n_layers+1, d_model] at the final prompt token.
    hook_fn: optional callable(start_idx, end_idx) -> context manager, used by steering.
    """
    n = len(df)
    n_hidden = lm.model.config.num_hidden_layers + 1
    d = lm.model.config.hidden_size
    layers = layers if layers is not None else list(range(n_hidden))
    acts = np.zeros((n, len(layers), d), dtype=np.float32) if cache else None
    logit_pos = np.zeros(n, dtype=np.float32)
    logit_neg = np.zeros(n, dtype=np.float32)
    prompts = [build_prompt(lm, q, f) for q, f in zip(df.question, df.answer_format)]
    fmts = list(df.answer_format)
    ids_cache = {f: answer_token_ids(lm, f) for f in set(fmts)}

    it = range(0, n, batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=math.ceil(n / batch_size))
        except ImportError:
            pass
    for s in it:
        e = min(s + batch_size, n)
        enc = lm.tok(prompts[s:e], return_tensors="pt", padding=True, add_special_tokens=not lm.is_chat)
        enc = {k: v.to(lm.device) for k, v in enc.items()}
        ctx = hook_fn(s, e) if hook_fn else _null()
        with ctx:
            out = lm.model(**enc, output_hidden_states=cache, use_cache=False)
        last = out.logits[:, -1, :]  # left padding -> last position is the answer slot
        for j in range(e - s):
            p_ids, n_ids = ids_cache[fmts[s + j]]
            logit_pos[s + j] = _logsumexp(last[j:j + 1], p_ids).item()
            logit_neg[s + j] = _logsumexp(last[j:j + 1], n_ids).item()
        if cache:
            hs = torch.stack([out.hidden_states[L][:, -1, :] for L in layers], dim=1)
            acts[s:e] = hs.float().cpu().numpy()
        del out
    margin = logit_pos - logit_neg  # >0 means the model says Yes/True
    p_pos = 1.0 / (1.0 + np.exp(-margin))
    report = (margin > 0).astype(int)
    cols = {"logit_margin": margin, "p_pos": p_pos, "report": report}
    return cols, acts


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def get_blocks(model):
    """Return the list of transformer blocks for common HF architectures."""
    for path in ("model.layers", "gpt_neox.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Unknown architecture: cannot find transformer blocks")
