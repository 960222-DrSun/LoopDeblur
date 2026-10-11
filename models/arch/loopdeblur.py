"""LoopDeblur: adaptive recurrent image deblurring with disentangled
heterogeneous blur.

Paper: "LoopDeblur: Adaptive Recurrent Image Deblurring with Disentangled
Heterogeneous Blur" (ICASSP).

Core equations (Sec. 2.2 - 2.3):

    P          = E(Y) = W_in * Y,                       (1)  input embedding, once
    S_0        = 0
    (a_t, r_t) = G_phi(S_t, P, t),  a_t = softmax(at),  (2)  degradation gate
                                    r_t = sigmoid(ct)
    S_{t+1}    = sum_i a_{t,i} * M_theta_i(S_t + P)     (3)  gated state update
    X_{t+1}    = clip(Y + D(S_{t+1}))                   (4)  residual decoding
    b(Y)       = min({t + 1 : r_t < eta} U {B_max})     (5)  adaptive termination

Key design points:
  * ``P`` is computed **once** and re-injected before every state update, which
    anchors each iteration to the original blurred input and prevents
    input-independent fixed-point collapse.
  * The module library ``{M_theta_i}`` is instantiated **once** and shared across
    all recurrent steps; the parameter count therefore depends only on the
    library size ``K``, not on the iteration count.
  * ``alpha_t`` encodes the *type* of correction; ``b(Y)`` encodes the *severity*.
    This is the type/severity disentanglement.
  * Execution modes: ``soft`` (differentiable mixture of all K modules, used in
    training) and ``hard`` (top-1 only, 1/K the per-step cost, used at inference).
    The measured gap is -0.48 dB (paper Table 3, row h).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .restormer_blocks import RestormerBlock, LayerNorm

__all__ = ['LoopDeblur', 'DegradationGate', 'DegradationModule']


class DegradationModule(nn.Module):
    """One entry ``M_theta_i`` of the degradation-module library.

    The framework is **backbone-agnostic**: ``M_theta_i`` may be a CNN, a
    Transformer, a state-space or a hybrid operator. Following the paper, the
    default instantiation stacks ``M`` lightweight pre-normalised
    Restormer blocks (MDTA + GDFN), whose transposed channel attention keeps the
    per-head cost at ``O(d_h^2 * H * W)`` rather than ``O((HW)^2 * d_h)``.
    """

    def __init__(self, dim, num_blocks=2, num_heads=4):
        super().__init__()
        self.body = nn.Sequential(*[
            RestormerBlock(dim, num_heads=num_heads) for _ in range(num_blocks)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return x + self.norm(self.body(x))


class DegradationGate(nn.Module):
    """Degradation-gated controller ``G_phi(S_t, P, t)``.

    Predicts the two control signals of Eq. (2):

    * ``alpha_t`` -- softmax over the ``K`` correction modules; which *type* of
      correction is applied.
    * ``rho_t``   -- sigmoid continuation probability; whether to keep iterating,
      i.e. how much *severity* remains.

    The iteration index ``t`` is injected as a learned sinusoidal embedding so
    that the same gate can adopt a different policy at different depths.
    """

    def __init__(self, dim, num_modules, hidden=None):
        super().__init__()
        hidden = hidden or dim
        in_dim = dim * 2 + dim  # S_t, P, t-embedding
        self.state_proj = nn.Conv2d(dim, hidden, 1, bias=False)
        self.emb_proj = nn.Conv2d(dim, hidden, 1, bias=False)
        self.time_proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.trunk = nn.Sequential(
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(),
        )
        self.alpha_head = nn.Linear(hidden, num_modules)
        self.rho_head = nn.Linear(hidden, 1)

        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.rho_head.weight)
        # start with a high continuation probability so the recurrence runs long
        nn.init.constant_(self.rho_head.bias, 1.0)

    def _time_embedding(self, t, dim, device, dtype):
        """Sinusoidal embedding of the iteration index."""
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device,
                                                             dtype=dtype) / max(half - 1, 1))
        arg = t * freqs
        emb = torch.cat([torch.sin(arg), torch.cos(arg)])
        if emb.numel() < dim:
            emb = F.pad(emb, (0, dim - emb.numel()))
        return emb

    def forward(self, state, embedding, t):
        b, d, h, w = state.shape
        feat = self.state_proj(state) + self.emb_proj(embedding)
        t_emb = self._time_embedding(torch.tensor(float(t)), d, state.device,
                                     state.dtype).view(1, d, 1, 1)
        feat = feat + self.time_proj(t_emb.flatten(1)).view(1, -1, 1, 1)

        pooled = self.trunk(feat).mean(dim=(2, 3))  # (B, hidden)
        alpha = F.softmax(self.alpha_head(pooled), dim=-1)   # (B, K)
        rho = torch.sigmoid(self.rho_head(pooled)).squeeze(-1)  # (B,)
        return alpha, rho, alpha.new_zeros(alpha.shape)


@ARCH_REGISTRY.register()
class LoopDeblur(nn.Module):
    """Adaptive recurrent deblurring network.

    Args:
        in_channels (int): image channels. Default: 3.
        dim (int): feature width ``d``. Default: 48.
        num_modules (int): ``K``, degradation-module library size. **K=5 is the
            optimum reported in the paper** (0.29M / 1.46M / 2.32M parameters and
            27.06 / 34.67 / 33.72 dB for K = 1 / 5 / 8).
        blocks_per_module (int): ``M`` Restormer blocks inside each module.
        num_heads (int): attention heads inside each block.
        max_iters (int): ``B_max``, the maximum computational budget.
        threshold (float): ``eta``, continuation threshold for Eq. (5).
        hard_routing (bool): top-1 inference routing (1/K per-step cost).
        inference_budget (int | None): if set, force a fixed number of iterations
            ``B`` instead of adaptive stopping (used for the efficiency table).
        rand_start (float): probability of randomly truncating the forward pass
            during training, which keeps the gate's stopping policy calibrated.
        input_injection (bool): re-inject the embedding ``P`` at every step.
            Disabling it (ablation (f)) degenerates into an autonomous recurrent
            update and drops PSNR by 6.25 dB.
        residual_decode (bool): decode the residual ``X = Y + D(S)`` instead of
            regressing the image directly (ablation (g)). Residual decoding gives
            an identity-mapping initialisation and stabilises early training;
            direct regression loses 0.99 dB.
    """

    def __init__(self, in_channels=3, dim=48, num_modules=5, blocks_per_module=2,
                 num_heads=4, max_iters=6, threshold=0.5, hard_routing=False,
                 inference_budget=None, rand_start=0.3,
                 input_injection=True, residual_decode=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.K = num_modules
        self.max_iters = max_iters
        self.threshold = threshold
        self.hard_routing = hard_routing
        self.inference_budget = inference_budget
        self.rand_start = rand_start
        self.input_injection = input_injection
        self.residual_decode = residual_decode

        # ---- encoder: projects the blurry image into the fixed embedding P ----
        self.encoder = nn.Conv2d(in_channels, dim, 3, padding=1, bias=False)

        # ---- module library: instantiated ONCE, shared across all steps -------
        self.module_library = nn.ModuleList([
            DegradationModule(dim, num_blocks=blocks_per_module, num_heads=num_heads)
            for _ in range(num_modules)
        ])

        # ---- degradation gate -------------------------------------------------
        self.gate = DegradationGate(dim, num_modules)

        # ---- residual decoder D: dim -> in_channels (Eq. 4) -------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, in_channels, 1, bias=False),
        )
        nn.init.zeros_(self.decoder[-1].weight)  # identity mapping at init

        # per-step diagnostics, filled during forward
        self.last_alphas = []
        self.last_rhos = []

    # ------------------------------------------------------------------
    def _apply_library(self, x, alpha):
        """Eq. (3): soft mixture (training) or hard top-1 (inference)."""
        if self.hard_routing and not self.training:
            idx = int(torch.argmax(alpha, dim=-1)[0].item())
            return self.module_library[idx](x), idx

        stacked = torch.stack([m(x) for m in self.module_library], dim=1)
        w = alpha.view(alpha.shape[0], self.K, 1, 1, 1)
        return (stacked * w).sum(dim=1), None

    # ------------------------------------------------------------------
    def forward(self, x, num_iters=None, return_trajectory=False):
        """Args:
            x (Tensor): blurry image, (B, 3, H, W) in [0, 1].
            num_iters (int | None): force a fixed iteration count (overrides the
                adaptive stopping rule). Used for the efficiency evaluation.
            return_trajectory (bool): also return ``X_1..X_b(Y)``.

        Returns:
            Tensor or (Tensor, list[Tensor]).
        """
        # P is computed ONCE and re-injected at every step (Eq. 1)
        embedding = self.encoder(x)
        state = torch.zeros_like(embedding)  # S_0 = 0

        budget = num_iters if num_iters is not None else self.max_iters
        if self.inference_budget is not None and not self.training and num_iters is None:
            budget = self.inference_budget

        trajectory = []
        self.last_alphas, self.last_rhos = [], []

        out = x
        for t in range(self.max_iters):
            alpha, rho, gate_logits = self.gate(state, embedding, t)

            # gated state update, conditioned on S_t + P (Eq. 3).
            # Ablation (f) removes the embedding injection entirely, so the
            # recurrence becomes autonomous and drifts away from the input.
            delta = state + embedding if self.input_injection else state
            updated, _ = self._apply_library(delta, alpha)

            # ---- adaptive early exit (inference only) -----------------------
            if not self.training and num_iters is None:
                # Eq. (5): stop as soon as the continuation probability drops.
                # Guarded by ``t > 0`` so at least one update is always applied.
                if bool((rho < self.threshold).all()) and t > 0:
                    state = 0.5 * (state + updated)
                    out = self._decode(x, state)
                    trajectory.append(out)
                    self.last_alphas.append(alpha)
                    self.last_rhos.append(rho)
                    break

            # keep the state on the same scale every step: a plain additive
            # recurrence would make the activations grow linearly with t, so the
            # update is written as a residual around the state itself.
            state = updated if t == 0 else 0.5 * (state + updated)
            out = self._decode(x, state)  # residual decoding (Eq. 4)
            trajectory.append(out)
            self.last_alphas.append(alpha)
            self.last_rhos.append(rho)

            if t + 1 >= budget:
                break

        if return_trajectory:
            return out, trajectory
        return out

    def _decode(self, x, state):
        """Eq. (4): ``X = clip(Y + D(S))``.

        Ablation (g) replaces the residual form with direct regression.
        """
        pred = self.decoder(state)
        if self.residual_decode:
            pred = x + pred
        return pred.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def gate_statistics(self):
        """Returns ``(alphas, rhos)`` from the most recent forward pass.

        ``alphas`` is a list of (B, K) tensors and ``rhos`` a list of (B,)
        tensors, one entry per executed iteration. Consumed by
        :class:`~models.losses.adaptive_regularisation.AdaptiveGateLoss`.
        """
        return self.last_alphas, self.last_rhos
