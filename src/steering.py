"""Activation steering for the causal-access test (H4).

Idea: if the probe's internal "truth" signal is merely decodable but unused,
amplifying it should not change the report. If it is *accessible*, pushing the
residual stream further in the direction the probe already points should flip
wrong answers to right ones.

Crucially, the push uses the PROBE'S OWN prediction (sign of its out-of-fold
score), never the ground-truth label. So steering can only fix an error when the
probe itself was right: this tests access to internal information, not
injection of the answer.

hidden_states[L] in Hugging Face is the output of block L-1, so a direction
measured at hidden_states[L] is added to the output of block L-1, at every token
position.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from .model import get_blocks


def make_hook_factory(lm, layer: int, direction: np.ndarray, signs: np.ndarray, scale: float):
    """Return hook_fn(start, end) -> context manager, for model.run_items.

    direction : unit vector [d_model] (numpy)
    signs     : per-item +1/-1 (which way to push each item), length n_items
    scale     : absolute norm of the added vector
    """
    if layer < 1:
        raise ValueError("Cannot steer at the embedding layer (layer 0)")
    block = get_blocks(lm.model)[layer - 1]
    p = next(block.parameters())
    vec = torch.tensor(direction, dtype=p.dtype, device=p.device) * scale

    @contextmanager
    def hook_fn(s, e):
        sg = torch.tensor(signs[s:e], dtype=p.dtype, device=p.device).view(-1, 1, 1)

        def fwd_hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                return (h + sg.to(h.device) * vec.to(h.device),) + tuple(output[1:])
            return output + sg.to(output.device) * vec.to(output.device)

        handle = block.register_forward_hook(fwd_hook)
        try:
            yield
        finally:
            handle.remove()

    return hook_fn


def random_unit_like(direction: np.ndarray, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(direction.shape).astype(np.float32)
    return v / np.linalg.norm(v)
