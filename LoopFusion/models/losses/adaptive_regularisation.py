"""Adaptive-computation losses for LoopDeblur.

Paper: "LoopDeblur: Adaptive Recurrent Image Deblurring with Disentangled
Heterogeneous Blur" (ICASSP), Sec. 2.5, Eq. (6) and Eq. (7).

Deep supervision (Eq. 6):

    L_rec = ||X_{B_max} - X_gt||_1
            + lambda * sum_{t in T_tail, t < B_max} ||X_t - X_gt||_1

with ``lambda = 0.5``. Only the last ``T`` tracked iterations carry gradient;
supervising the whole trajectory stabilises the recurrence and prevents it from
collapsing into a single-step solution (ablating it costs 1.81 dB, Table 3 row e).

Computation-aware regularisation (Eq. 7):

    L = L_rec + mu * sum_t rho_t + nu * sum_t H(alpha_t)

* ``mu * sum_t rho_t``      -- continuation penalty: pushes ``rho_t`` down, so
  fewer iterations are executed at inference. Ablating it costs 0.29 dB
  (Table 3 row d) but saves computation.
* ``nu * sum_t H(alpha_t)`` -- entropy regularisation on the module mixture:
  drives the assignment toward determinism, which makes the hard top-1
  inference routing a faithful approximation of the soft training mixture.
  Ablating it costs 0.36 dB (Table 3 row c).

Both terms are **not** accuracy terms: they trade a small amount of PSNR for
substantially cheaper inference (74.8 vs 121.2 GMACs, about -38%).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY

__all__ = ['DeepSupervisionLoss', 'AdaptiveGateLoss', 'LoopDeblurLoss']


@LOSS_REGISTRY.register()
class DeepSupervisionLoss(nn.Module):
    """Deep supervision over the last ``T`` tracked iterations (Eq. 6).

    Args:
        loss_weight (float): ``lambda`` in Eq. (6). The paper uses 0.5.
        criterion (str): base distance, 'l1' (paper) or 'l2'.
    """

    def __init__(self, loss_weight=0.5, criterion='l1', **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = criterion

    def _dist(self, pred, gt):
        if self.criterion == 'l2':
            return F.mse_loss(pred, gt)
        return F.l1_loss(pred, gt)

    def forward(self, trajectory, gt, **kwargs):
        """Args:
            trajectory (list[Tensor]): ``[X_1, ..., X_b(Y)]``, coarse to fine.
            gt (Tensor): ground-truth sharp image.
        """
        if not trajectory:
            return gt.new_zeros(())
        # final iteration: always supervised
        loss = self._dist(trajectory[-1], gt)
        # intermediate tracked iterations (all but the last)
        if self.loss_weight > 0 and len(trajectory) > 1:
            tail = sum(self._dist(x, gt) for x in trajectory[:-1]) / (len(trajectory) - 1)
            loss = loss + self.loss_weight * tail
        return loss


@LOSS_REGISTRY.register()
class AdaptiveGateLoss(nn.Module):
    """Computation-aware gate regularisation: continuation + entropy (Eq. 7).

    Args:
        mu (float): weight of ``sum_t rho_t`` (continuation penalty). The paper
            reports the ablation with/without it; a reasonable default is 1e-3.
        nu (float): weight of ``sum_t H(alpha_t)`` (entropy regularisation).
    """

    def __init__(self, mu=1e-3, nu=1e-3, **kwargs):
        super().__init__()
        self.mu = mu
        self.nu = nu

    def forward(self, alphas, rhos, **kwargs):
        """Args:
            alphas (list[Tensor]): per-iteration (B, K) mixture weights.
            rhos (list[Tensor]): per-iteration (B,) continuation probabilities.
        """
        if not alphas:
            return self.mu * 0.0 + self.nu * 0.0

        loss = torch.zeros((), dtype=alphas[0].dtype, device=alphas[0].device)

        if self.mu > 0 and rhos:
            # sum_t rho_t  -> minimise means "stop earlier"
            loss = loss + self.mu * torch.stack([r.mean() for r in rhos]).sum()

        if self.nu > 0:
            # H(alpha_t) = -sum_i alpha_i log alpha_i ; minimising it sharpens the
            # routing so that hard top-1 inference matches the soft training mix.
            ent = []
            for a in alphas:
                ent.append(-(a * torch.log(a + 1e-8)).sum(dim=-1).mean())
            loss = loss + self.nu * torch.stack(ent).sum()

        return loss


@LOSS_REGISTRY.register()
class LoopDeblurLoss(nn.Module):
    """Composite objective ``L = L_rec + mu*sum rho + nu*sum H(alpha)`` (Eq. 7).

    Convenience wrapper that keeps the whole objective in a single module so it
    can be declared directly in the yml:

        train:
          loop_opt:
            type: LoopDeblurLoss
            lambda_ds: 0.5     # deep supervision weight
            mu: 0.001          # continuation penalty
            nu: 0.001          # entropy regularisation
    """

    def __init__(self, lambda_ds=0.5, mu=1e-3, nu=1e-3, criterion='l1', **kwargs):
        super().__init__()
        self.ds = DeepSupervisionLoss(loss_weight=lambda_ds, criterion=criterion)
        self.gate = AdaptiveGateLoss(mu=mu, nu=nu)
        self.loss_dict = {}

    def forward(self, trajectory, gt, alphas=None, rhos=None, **kwargs):
        l_rec = self.ds(trajectory, gt)
        l_gate = self.gate(alphas or [], rhos or [])
        self.loss_dict = {
            'l_rec': float(l_rec.detach()),
            'l_gate': float(l_gate.detach()) if torch.is_tensor(l_gate) else float(l_gate),
        }
        return l_rec + l_gate
