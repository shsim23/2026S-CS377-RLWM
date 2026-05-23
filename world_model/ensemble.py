from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .single import SingleWorldModel
from .utils import symexp, weight_init
from .constants import NUM_ENSEMBLE_MEMBERS, LATENT_DIM, GRU_HIDDEN


class EnsembleWorldModel(nn.Module):
    def __init__(self, num_members: int = NUM_ENSEMBLE_MEMBERS, **kwargs):
        super().__init__()
        self.K = num_members
        self.members = nn.ModuleList([SingleWorldModel(**kwargs) for _ in range(num_members)])

        for k, m in enumerate(self.members):
            torch.manual_seed(42 + k * 1000)
            m.apply(weight_init)
        torch.manual_seed(42)

        self.latent_dim = self.members[0].latent_dim
        self.gru_hidden = self.members[0].gru_hidden

    # ------------------------------------------------------------------ #
    def encode(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zs = torch.stack([m.encoder(s) for m in self.members], dim=1)
        z_mean = zs.mean(dim=1)
        h_init = torch.zeros(s.shape[0], self.gru_hidden, device=s.device)
        return z_mean, h_init

    def warmup_h(
        self,
        prefix_states: torch.Tensor,
        prefix_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = prefix_states.shape
        z_final_list, h_final_list = [], []
        for m in self.members:
            z_all = m.encoder(prefix_states)
            h = torch.zeros(B, self.gru_hidden, device=prefix_states.device)
            for t in range(P - 1):
                a_emb = m.action_embedder(prefix_actions[:, t])
                _, h = m.dynamics(z_all[:, t], a_emb, h)
            z_final_list.append(z_all[:, -1])
            h_final_list.append(h)
        z_final = torch.stack(z_final_list, dim=1).mean(dim=1)
        h_final = torch.stack(h_final_list, dim=1).mean(dim=1)
        return z_final, h_final

    @torch.no_grad()
    def imagine_step(
        self, z: torch.Tensor, h: torch.Tensor, a: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_nexts, h_nexts, r_symlogs, d_probs = [], [], [], []
        for m in self.members:
            out = m.imagine_step(z, h, a)
            z_nexts.append(out["z_next"])
            h_nexts.append(out["h_next"])
            r_symlogs.append(out["reward_symlog"])
            d_probs.append(out["done"])

        z_stack = torch.stack(z_nexts, dim=1)
        h_stack = torch.stack(h_nexts, dim=1)
        r_stack = torch.stack(r_symlogs, dim=1)
        d_stack = torch.stack(d_probs, dim=1)

        z_var_per_dim = z_stack.var(dim=1)
        sigma = torch.sqrt(z_var_per_dim.mean(dim=-1) + 1e-8)

        return {
            "z_next": z_stack.mean(dim=1),
            "h_next": h_stack.mean(dim=1),
            "reward": symexp(r_stack.mean(dim=1)),
            "done":   d_stack.mean(dim=1),
            "sigma":  sigma,
        }

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "K": self.K}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "EnsembleWorldModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(num_members=ckpt["K"], **kwargs)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
