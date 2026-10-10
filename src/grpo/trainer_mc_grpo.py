"""
MC-GRPO: Mode-Length-Normalized advantage.

A variant of GRPO. Standard GRPO uses the group-normalized advantage:

    A_i = (r_i - mu_g) / sigma_g

MC-GRPO additionally rescales it by a per-mode completion-length factor:

    A_i = (r_i - mu_g) / sigma_g  *  (L_bar / L_m)

where:
  - L_m    : mean completion length (in tokens) of reasoning mode m in the batch
  - L_bar  : mean of the three per-mode mean lengths
  - m(i)   : reasoning mode (low / medium / high) of sample i
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer


class MCGRPOTrainer(GRPOTrainer):
    """GRPO with per-mode completion-length advantage normalization."""

    _MODES = ("low", "medium", "high")

    def _generate_and_score_completions(self, inputs):
        # Base class: rollouts, rewards, group-normalized advantages (scale_rewards="group" -> A = (r - mu_g)/sigma_g).
        output = super()._generate_and_score_completions(inputs)

        advantages = output.get("advantages")
        cmask = output.get("completion_mask")
        if not isinstance(advantages, torch.Tensor) or advantages.numel() == 0:
            return output
        if not isinstance(cmask, torch.Tensor) or cmask.shape[0] != advantages.shape[0]:
            return output

        # inputs is list[dict]; advantages/cmask are sliced to this process and have length == len(inputs). Mode labels align 1:1.
        modes = [ex.get("mode", "low") for ex in inputs]
        if len(modes) != advantages.shape[0]:
            return output

        # Per-completion length (number of non-pad completion tokens).
        lengths = cmask.sum(dim=1).float()

        per_mode_idx = {
            m: [i for i, mm in enumerate(modes) if mm == m] for m in self._MODES
        }
        mode_len = {
            m: lengths[idx].mean().item()
            for m, idx in per_mode_idx.items()
            if len(idx) >= 1
        }
        if len(mode_len) < 2:  # Need >=2 modes present to stratify
            return output

        eps = 1e-8
        mean_len = sum(mode_len.values()) / len(mode_len)
        scaled = advantages.clone()
        for m, L_m in mode_len.items():
            idx = per_mode_idx[m]
            scaled[idx] = scaled[idx] * (mean_len / (L_m + eps))
        output["advantages"] = scaled

        # Prevention for crash training 
        try:
            te = "train" if self.model.training else "eval"
            for m, L_m in mode_len.items():
                idx = per_mode_idx[m]
                self._metrics[te][f"completion_length/{m}"].append(L_m)
                self._metrics[te][f"mc_grpo/len_factor_{m}"].append(
                    mean_len / (L_m + eps)
                )
                # trial-and-error for average std in advantage
                if len(idx) >= 2:
                    self._metrics[te][f"mc_grpo/adv_std_{m}"].append(
                        advantages[idx].std().item()
                    )
            self._metrics[te]["mc_grpo/modes_present"].append(float(len(mode_len)))
        except Exception:
            pass

        return output
