# LoopDeblur

**LoopDeblur: Adaptive Recurrent Image Deblurring with Disentangled Heterogeneous Blur**
---
Overview of LoopDeblur
![alt text](image.png)


## Project Layout

```
LoopFusion/
├── models/
│   ├── arch/
│   │   ├── loopdeblur.py            # LoopDeblur + DegradationGate + DegradationModule
│   │   └── restormer_blocks.py      # MDTA + GDFN blocks used inside each module
│   ├── data/
│   │   └── paired_deblur_dataset.py # GoPro / HIDE / RealBlur paired dataset
│   ├── losses/
│   │   └── adaptive_regularisation.py  # Eq. (6) deep supervision + Eq. (7) gate reg.
│   └── loopdeblur_model.py          # SRModel subclass, truncated BPTT pipeline
├── options/
│   ├── train/LoopDeblur_GoPro.yml
│   ├── train/LoopDeblur_GoPro_Ablation.yml
│   └── test/LoopDeblur_GoPro.yml
├── train.py
├── inference_loopdeblur.py          # reports b(Y) per image
└── requirements.txt
```

---

## Install

BasicSR is a **real external dependency**; the local `models` package registers
everything into the official registries on import.

```bash
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python setup.py develop     # or: pip install -e .
```

> **Checkpoint format.** In this file the options are stored as nested Python dicts
> in ``alpha``/``rho`` lists; use ``--force_yml`` for overrides rather than editing
> the yml when running the ablation grid.

---

## Training

```bash
python train.py -opt options/train/LoopDeblur_GoPro.yml

# data paths without editing the yml
python train.py -opt options/train/LoopDeblur_GoPro.yml \
    --force_yml dataroot_gt=/data/GoPro/train/sharp \
                dataroot_lq=/data/GoPro/train/blur

# 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
    train.py -opt options/train/LoopDeblur_GoPro.yml --launcher pytorch
```

Paper hyper-parameters: `dim=48`, `K=5` modules, `M=2` Restormer blocks per module,
`B_max=6`, AdamW `lr=2e-4`, batch 8, `gt_size=256`, EMA 0.999, `lambda=0.5`,
`mu=nu=1e-3`.

### Truncated BPTT

`train.track_iters` (=`T`) and `train.warmup_iters` (=`B_max - T`) control the
memory/compute trade-off described in Sec. 2.5:

1. **Warm-up** — early iterations run under `no_grad`, approximating an identity
   mapping with faint gradient signals.
2. **Tracking window** — the final `T` iterations build the differentiation graph.
3. **Deep supervision** — every tracked iteration is supervised (Eq. 6).

This is what keeps memory bounded: without truncation, activations grow linearly
with `B_max`.

---

## Evaluation & inference

```bash
python basicsr/test.py -opt options/test/LoopDeblur_GoPro.yml

# adaptive stopping, prints mean/min/max b(Y)
python inference_loopdeblur.py \
    --checkpoint experiments/LoopDeblur_GoPro/models/net_g_latest.pth \
    --input datasets/GoPro/test/blur --output results/deblur

# fixed B=6 efficiency baseline
python inference_loopdeblur.py --checkpoint <ckpt> --input <dir> --fixed_iters 6

# lossless soft mixture at inference (+0.48 dB, K times slower)
python inference_loopdeblur.py --checkpoint <ckpt> --input <dir> --soft_routing
```

---

