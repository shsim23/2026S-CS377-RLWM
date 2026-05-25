import torch
import torch.nn.functional as F

from .utils import symlog, variance_regularization

# State-vector layout (matches pacman_env.state.StateBuilder, total 901 dim):
#   [0:2]      pacman pos
#   [2:18]     ghost slots (4 ghosts × 4 fields)
#   [18:459]   food mask (441 = 21×21 padded grid, binary)
#   [459:900]  wall mask (constant per episode — skipped in aux loss)
#   [900:901]  power timer
# DynamicStateHead predicts everything except walls → 460 output dims, slices:
#   [0:2], [2:18], [18:459], [459:460]
_PAC_SLICE   = slice(0, 2)
_GHOST_SLICE = slice(2, 18)
_FOOD_SLICE  = slice(18, 459)
_POWER_SLICE = slice(900, 901)   # in input state
_OUT_PAC_SLICE   = slice(0, 2)
_OUT_GHOST_SLICE = slice(2, 18)
_OUT_FOOD_SLICE  = slice(18, 459)
_OUT_POWER_SLICE = slice(459, 460)


def compute_world_model_loss(
    outputs: dict,
    rewards_raw: torch.Tensor,
    dones: torch.Tensor,
    states: torch.Tensor = None,
    beta_reward: float = 1.0,
    beta_done: float = 1.0,
    beta_var: float = 0.01,
    beta_dynamic_state: float = 0.0,
    beta_count_delta: float = 0.0,
    beta_food_eaten: float = 0.0,
    pos_weight_done: float = 1.0,
    target_std: float = 1.0,
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

    # Weighted BCE for done — counters the heavy class imbalance
    # (positive class ≈ 5% of steps in random-policy episodes).
    if pos_weight_done != 1.0:
        w = torch.where(d_target_s > 0.5,
                        torch.full_like(d_target_s, pos_weight_done),
                        torch.ones_like(d_target_s))
        L_done = F.binary_cross_entropy(
            d_pred_s.clamp(1e-7, 1 - 1e-7), d_target_s, weight=w,
        )
    else:
        L_done = F.binary_cross_entropy(d_pred_s.clamp(1e-7, 1 - 1e-7), d_target_s)

    L_var    = variance_regularization(z_all, target_std=target_std)

    def _dyn_component_losses(pred_dyn, target_states):
        """Compute per-component aux losses against a (B, T, 901) state slice."""
        l_pos   = ((pred_dyn[..., _OUT_PAC_SLICE]   - target_states[..., _PAC_SLICE])   ** 2).mean()
        l_ghost = ((pred_dyn[..., _OUT_GHOST_SLICE] - target_states[..., _GHOST_SLICE]) ** 2).mean()
        l_power = ((pred_dyn[..., _OUT_POWER_SLICE] - target_states[..., _POWER_SLICE]) ** 2).mean()
        food_target = (target_states[..., _FOOD_SLICE] > 0.5).float()
        food_logits = pred_dyn[..., _OUT_FOOD_SLICE]
        l_food  = F.binary_cross_entropy_with_logits(food_logits, food_target)
        return l_pos, l_ghost, l_food, l_power

    L_dyn = torch.tensor(0.0, device=z_all.device)
    L_dyn_pos = L_dyn_ghost = L_dyn_food = L_dyn_power = torch.tensor(0.0, device=z_all.device)
    L_dyn_pred = torch.tensor(0.0, device=z_all.device)
    if beta_dynamic_state > 0.0 and "dyn_state_preds" in outputs and states is not None:
        # ---- (a) encoder path: dyn_state(z_all) vs state[t] ----
        pred_enc = outputs["dyn_state_preds"][:, burnin:]
        st_enc = states[:, burnin:]
        L_dyn_pos, L_dyn_ghost, L_dyn_food, L_dyn_power = _dyn_component_losses(pred_enc, st_enc)
        L_dyn = L_dyn_pos + L_dyn_ghost + L_dyn_food + L_dyn_power

        # ---- (b) dynamics path: dyn_state(z_preds) vs state[t+1] (v8.2) ----
        # z_preds[t] predicts state[t+1] so target is states[:, 1:].
        if "dyn_state_z_preds" in outputs:
            pred_dyn = outputs["dyn_state_z_preds"][:, burnin:]
            st_dyn = states[:, burnin + 1: burnin + 1 + pred_dyn.shape[1]]
            l_pos2, l_ghost2, l_food2, l_power2 = _dyn_component_losses(pred_dyn, st_dyn)
            L_dyn_pred = l_pos2 + l_ghost2 + l_food2 + l_power2
            L_dyn = L_dyn + L_dyn_pred

    # ---- L_count_delta: explicit constraint that the sigmoid-sum delta
    # (which the deterministic reward depends on) matches the true food_eaten.
    # Without this, per-cell BCE swamps the reward-gradient signal — see
    # scripts/verify_reward_limit.py: cells learn well (BCE~0.028) but the
    # sum-of-441-sigmoids accumulates ±4 calibration noise on the count,
    # which floods the 0/1 food_eaten signal.
    L_count_delta = torch.tensor(0.0, device=z_all.device)
    if (beta_count_delta > 0.0
            and "dyn_state_preds" in outputs
            and "dyn_state_z_preds" in outputs
            and states is not None):
        food_logits_enc = outputs["dyn_state_preds"][..., _OUT_FOOD_SLICE]
        food_logits_dyn = outputs["dyn_state_z_preds"][..., _OUT_FOOD_SLICE]
        count_pred_t    = torch.sigmoid(food_logits_enc).sum(dim=-1)        # (B, T)
        count_pred_next = torch.sigmoid(food_logits_dyn).sum(dim=-1)        # (B, T-1)

        food_mask = (states[..., _FOOD_SLICE] > 0.5).float()                # (B, T, 441)
        food_eaten_true = (food_mask[:, :-1] * (1 - food_mask[:, 1:])).sum(dim=-1)  # (B, T-1)

        delta_pred = count_pred_t[:, :-1] - count_pred_next                 # (B, T-1)
        # signed (no clamp) so sign-errors are penalised — matches what the
        # reward path actually consumes before clamp(0).
        delta_pred_s    = delta_pred     [:, burnin:]
        food_eaten_s    = food_eaten_true[:, burnin:]
        L_count_delta = ((delta_pred_s - food_eaten_s) ** 2).mean()

    # ---- L_food_eaten: BCE on FoodEatenHead's binary classifier (v10) ----
    L_food_eaten = torch.tensor(0.0, device=z_all.device)
    L_fe_enc = L_fe_dyn = torch.tensor(0.0, device=z_all.device)
    if (beta_food_eaten > 0.0
            and "food_eaten_logit_enc" in outputs
            and states is not None):
        food_mask = (states[..., _FOOD_SLICE] > 0.5).float()
        food_eaten_true = (food_mask[:, :-1] * (1 - food_mask[:, 1:])).sum(dim=-1)
        food_eaten_bin  = (food_eaten_true >= 1.0).float()            # (B, T-1)

        fe_logit_enc = outputs["food_eaten_logit_enc"][:, burnin:]
        fe_logit_dyn = outputs["food_eaten_logit_dyn"][:, burnin:]
        fe_target    = food_eaten_bin[:, burnin:]

        L_fe_enc = F.binary_cross_entropy_with_logits(fe_logit_enc, fe_target)
        L_fe_dyn = F.binary_cross_entropy_with_logits(fe_logit_dyn, fe_target)
        L_food_eaten = L_fe_enc + L_fe_dyn

    loss = (L_latent
            + beta_reward * L_reward
            + beta_done * L_done
            + beta_var * L_var
            + beta_dynamic_state * L_dyn
            + beta_count_delta * L_count_delta
            + beta_food_eaten * L_food_eaten)

    return loss, {
        "L_total":        loss.item(),
        "L_latent":       L_latent.item(),
        "L_reward":       L_reward.item(),
        "L_done":         L_done.item(),
        "L_var":          L_var.item(),
        "L_dyn":          L_dyn.item(),
        "L_dyn_pos":      L_dyn_pos.item(),
        "L_dyn_ghost":    L_dyn_ghost.item(),
        "L_dyn_food":     L_dyn_food.item(),
        "L_dyn_power":    L_dyn_power.item(),
        "L_dyn_pred":     L_dyn_pred.item(),
        "L_count_delta":  L_count_delta.item(),
        "L_food_eaten":   L_food_eaten.item(),
        "L_fe_enc":       L_fe_enc.item(),
        "L_fe_dyn":       L_fe_dyn.item(),
    }
