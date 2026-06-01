"""LaProp optimizer + adaptive gradient clipping (spec §7).

DreamerV3 uses LaProp (RMSProp-style normalization applied *before* momentum,
unlike Adam which momentums the raw gradient then normalizes) together with
adaptive gradient clipping (AGC), which clips each parameter's gradient by the
ratio of its grad-norm to its weight-norm rather than a single global threshold.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class LaProp(Optimizer):
    """LaProp (Ziyin et al., 2020): normalize the gradient by its RMS *first*,
    then accumulate momentum on the normalized gradient.

        v_t = β2 v_{t-1} + (1−β2) g^2
        m_t = β1 m_{t-1} + (1−β1) g / (sqrt(v_t) + eps)
        θ  -= lr * m_t / (1 − β1^t)          (with bias correction)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                v_hat = v / (1 - b2 ** t)
                normed = g / (v_hat.sqrt() + eps)
                m.mul_(b1).add_(normed, alpha=1 - b1)
                m_hat = m / (1 - b1 ** t)
                p.add_(m_hat, alpha=-lr)
        return loss


@torch.no_grad()
def adaptive_grad_clip(parameters, clip: float = 0.3, eps: float = 1e-3) -> None:
    """Adaptive gradient clipping (Brock et al., 2021).

    For each parameter, scale its gradient down so that
        ||g|| <= clip * max(||θ||, eps).
    Applied in-place before `optimizer.step()`."""
    for p in parameters:
        if p.grad is None:
            continue
        p_norm = p.detach().norm()
        g_norm = p.grad.detach().norm()
        max_norm = clip * torch.clamp(p_norm, min=eps)
        if g_norm > max_norm:
            p.grad.mul_(max_norm / (g_norm + 1e-6))
