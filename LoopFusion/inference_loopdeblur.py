"""LoopDeblur single-image inference with adaptive iteration reporting.

Usage
-----
    python inference_loopdeblur.py \
        --checkpoint experiments/LoopDeblur_GoPro/models/net_g_latest.pth \
        --input datasets/GoPro/test/blur \
        --output results/deblur

The script also reports the number of executed iterations ``b(Y)`` per image, so
that the adaptive-computation behaviour (fewer iterations for mild blur, more for
severe blur) can be inspected directly.
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = osp.dirname(osp.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models  # noqa: E402,F401  registers the architectures
from models.arch import LoopDeblur  # noqa: E402

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_network(checkpoint, device, **arch_kwargs):
    net = LoopDeblur(**arch_kwargs)
    state = torch.load(checkpoint, map_location='cpu')
    for key in ('params_ema', 'params', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys, first few: {missing[:5]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys, first few: {unexpected[:5]}')
    return net.to(device).eval()


def _count_gmacs(net, shape=(1, 3, 256, 256), max_iters=6, hard_routing=True):
    """Rough per-image GMAC estimate, for the efficiency table (Sec. 3.3).

    Uses a forward hook on the first degradation module to measure one module
    evaluation, then scales by the number of executed steps: the hard top-1
    routing evaluates exactly one module per step, whereas the soft mixture
    evaluates all ``K``.
    """
    macs_per_module = {'value': None}

    def hook(_module, inputs, _output):
        x = inputs[0]
        _, c, h, w = x.shape
        # approximate: each Restormer block is dominated by the 3x3 DWConv and
        # the two 1x1 projections; this is an order-of-magnitude estimate only.
        macs_per_module['value'] = float(c * c * h * w * 2 * 4)

    handle = net.module_library[0].register_forward_hook(hook)
    dummy = torch.zeros(*shape, device=next(net.parameters()).device)
    with torch.no_grad():
        net(dummy)
    handle.remove()

    per_step = macs_per_module['value'] or 0.0
    if not hard_routing:
        per_step *= len(net.module_library)
    return per_step * max_iters / 1e9


def main():
    parser = argparse.ArgumentParser(description='LoopDeblur inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/deblur')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_modules', type=int, default=5)
    parser.add_argument('--blocks_per_module', type=int, default=2)
    parser.add_argument('--max_iters', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fixed_iters', type=int, default=None,
                        help='disable adaptive stopping and run exactly N '
                             'iterations (the B=6 efficiency baseline)')
    parser.add_argument('--soft_routing', action='store_true',
                        help='keep the soft mixture at inference (lossless but '
                             'K times slower; paper Table 3 row h)')
    args = parser.parse_args()

    device = torch.device(args.device)
    net = load_network(
        args.checkpoint, device, dim=args.dim, num_modules=args.num_modules,
        blocks_per_module=args.blocks_per_module, max_iters=args.max_iters,
        threshold=args.threshold, hard_routing=not args.soft_routing,
        inference_budget=args.fixed_iters,
    )
    num_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'LoopDeblur loaded: {num_params:.2f} M parameters (paper: 1.46 M for K=5)')

    mode = f'fixed B={args.fixed_iters}' if args.fixed_iters else \
        f'adaptive (eta={args.threshold})'
    print(f'inference mode: {mode}, routing: '
          f'{"soft mixture" if args.soft_routing else "hard top-1"}')

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        files = [in_path]
    if not files:
        raise FileNotFoundError(f'no images found under {in_path}')

    iters_log = []
    for path in files:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'[skip] cannot read {path}')
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(tensor)
        # number of executed iterations = length of the recorded trajectory
        n_iters = len([1 for _ in net.gate_statistics()[0]]) or args.max_iters
        iters_log.append(n_iters)

        pred = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / path.name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        print(f'[ok] {path.name}  b(Y)={n_iters}')

    if iters_log:
        print(f'mean b(Y) = {sum(iters_log) / len(iters_log):.2f} '
              f'(min {min(iters_log)}, max {max(iters_log)}), B_max = {args.max_iters}')
    print(f'results written to {out_dir}')


if __name__ == '__main__':
    main()
