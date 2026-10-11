"""LoopDeblur model wrapper.

Paper: "LoopDeblur: Adaptive Recurrent Image Deblurring with Disentangled
Heterogeneous Blur" (ICASSP), Sec. 2.5 "Training Pipeline".

The recurrence is trained with **truncated BPTT combined with incremental
checkpointing**, in three phases:

1. **Warm-up** (``B_max - T`` steps, no gradient). States propagate from ``S_0``
   to depth ``B_max - T`` under ``torch.no_grad()``. These early iterations
   approximate an identity mapping with faint gradient signals, so only forward
   computation is required — this is what keeps memory from growing linearly in
   the iteration count.
2. **Tracking window** (``T`` steps, with gradient). The state is ``detach()``-ed
   and the differentiation graph is built for the final ``T`` iterations, so
   gradients flow through ``T * M`` blocks for output prediction and gate
   optimisation. Incremental checkpointing discards the intermediate activations
   of the module mixture and recomputes them during backpropagation.
3. **Deep supervision and gate regularisation** — see ``models/losses``.

Total objective: ``L = L_rec + mu * sum_t rho_t + nu * sum_t H(alpha_t)``.
"""

import torch
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

from models.arch import LoopDeblur as LoopDeblurArch  # noqa: F401  ensures registration
from models.losses.adaptive_regularisation import DeepSupervisionLoss, AdaptiveGateLoss

__all__ = ['LoopDeblurModel']


@MODEL_REGISTRY.register()
class LoopDeblurModel(SRModel):
    """SRModel specialised for the gated recurrent LoopDeblur architecture.

    ``opt['train']['loop_opt']`` controls the adaptive objective::

        train:
          loop_opt:
            lambda_ds: 0.5     # deep supervision weight (Eq. 6)
            mu: 0.001          # continuation penalty   (Eq. 7)
            nu: 0.001          # entropy regularisation (Eq. 7)
          track_iters: 3       # T: tracked iterations carrying gradient
          warmup_iters: 6      # B_max - T
    """

    def init_training_settings(self):
        super().init_training_settings()

        loop_opt = self.opt['train'].get('loop_opt', {})
        self.track_iters = int(self.opt['train'].get('track_iters', 3))
        self.warmup_steps = int(self.opt['train'].get('warmup_iters', 0))
        self.ds_loss = DeepSupervisionLoss(
            loss_weight=loop_opt.get('lambda_ds', 0.5),
            criterion=loop_opt.get('criterion', 'l1'),
        ).to(self.device)
        self.gate_loss = AdaptiveGateLoss(
            mu=loop_opt.get('mu', 1e-3), nu=loop_opt.get('nu', 1e-3)
        ).to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    # ------------------------------------------------------------------
    def _recurrent_forward(self, lq, num_iters=None, return_trajectory=False):
        """Run the recurrence, returning the final estimate and the trajectory.

        The warm-up / tracking split is delegated to the architecture: the
        network exposes ``max_iters`` and we simply request a trajectory. Memory
        is bounded because the architecture re-injects a detached embedding and
        the gate is the only stateful coupling across steps.
        """
        return self.net_g(lq, num_iters=num_iters,
                          return_trajectory=return_trajectory)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        output, trajectory = self.net_g(self.lq, return_trajectory=True)
        self.output = output

        # ---- Eq. (6): deep supervision over the tracked trajectory --------
        l_rec = self.ds_loss(trajectory, self.gt)

        # ---- Eq. (7): continuation penalty + entropy regularisation -------
        alphas, rhos = self.net_g.gate_statistics()
        l_gate = self.gate_loss(alphas, rhos)

        l_total = l_rec + l_gate
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict({
            'l_rec': float(l_rec.detach()),
            'l_gate': float(l_gate.detach()) if torch.is_tensor(l_gate) else float(l_gate),
            'iters': float(len(trajectory)),
        })

    # ------------------------------------------------------------------
    def test(self):
        """Fixed-budget evaluation, mirroring the efficiency setting.

        Efficiency figures in the paper are obtained with **hard top-1 routing**
        at inference, so the per-step cost equals a single module rather than the
        full soft mixture.
        """
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    @torch.no_grad()
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, **kwargs):
        if self.opt.get('rank', 0) != 0:
            return None
        return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def adaptive_iteration_report(self, dataloader):
        """Diagnostic: mean executed iterations ``b(Y)`` per difficulty bucket.

        Useful for verifying that the type/severity disentanglement actually
        allocates fewer iterations to mild blur and more to severe blur.
        """
        self.net_g.eval()
        counts = []
        for data in dataloader:
            self.feed_data(data)
            _, trajectory = self.net_g(self.lq, return_trajectory=True)
            counts.append(len(trajectory))
        self.net_g.train()
        return {
            'mean_iters': float(sum(counts) / max(len(counts), 1)),
            'min_iters': int(min(counts)) if counts else 0,
            'max_iters': int(max(counts)) if counts else 0,
        }
