import torch
import torch.nn.functional as F

from .utils import symlog, variance_regularization


def compute_world_model_loss(
    outputs: dict,
    rewards_raw: torch.Tensor,
    dones: torch.Tensor,
    beta_reward: float = 1.0,
    beta_done: float = 1.0,
    beta_var: float = 0.01,
):
    burnin  = outputs["burnin"]
    z_all   = outputs["z_all"]
    z_preds = outputs["z_preds"]
    r_preds = outputs["r_preds"]
    d_preds = outputs["d_preds"]

    z_target = z_all[:, 1:].detach()
    r_target = symlog(rewards_raw[:, :-1])
    d_target = dones[:, :-1].float()

    z_pred_s   = z_preds[:, burnin:]
    z_target_s = z_target[:, burnin:]
    r_pred_s   = r_preds[:, burnin:]
    r_target_s = r_target[:, burnin:]
    d_pred_s   = d_preds[:, burnin:]
    d_target_s = d_target[:, burnin:]

    L_latent = ((z_pred_s - z_target_s) ** 2).mean()
    L_reward = ((r_pred_s - r_target_s) ** 2).mean()
    L_done   = F.binary_cross_entropy(d_pred_s.clamp(1e-7, 1 - 1e-7), d_target_s)
    L_var    = variance_regularization(z_all)

    loss = L_latent + beta_reward * L_reward + beta_done * L_done + beta_var * L_var

    return loss, {
        "L_total":  loss.item(),
        "L_latent": L_latent.item(),
        "L_reward": L_reward.item(),
        "L_done":   L_done.item(),
        "L_var":    L_var.item(),
    }
