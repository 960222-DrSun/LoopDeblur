"""Paired blur/sharp dataset for image deblurring.

Supports the four benchmarks used in the paper:
    GoPro, HIDE            -> synthetic motion blur
    RealBlur-R, RealBlur-J -> real camera blur (RAW / JPEG)

Expected on-disk layout::

    <dataroot>/<split>/blur/*.png
    <dataroot>/<split>/sharp/*.png

For RealBlur the original release ships ``blur``/``sharp`` (JPEG) plus a RAW
variant; point ``dataroot_gt``/``dataroot_lq`` at the corresponding folders.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


@DATASET_REGISTRY.register()
class PairedDeblurDataset(data.Dataset):
    """Paired blur/sharp dataset with patch cropping and flipping augmentation.

    Args:
        opt (dict): dataset options, see ``options/train/*.yml``.
            dataroot_gt   (str): folder of sharp images.
            dataroot_lq   (str): folder of blurry images (same file names).
            io_backend    (dict): e.g. {'type': 'disk'} or {'type': 'lmdb', ...}.
            gt_size       (int): crop size; 0 disables cropping.
            phase         (str): 'train' | 'val' | 'test'.
            scale         (int): spatial scale between gt and lq. Default: 1.
            num_frame     (int): unused, kept for API symmetry.
            use_flip      (bool): random horizontal flip.
            use_rot       (bool): random 90-degree rotation.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt.get('phase', 'train')
        self.gt_size = opt.get('gt_size', 256)
        self.scale = opt.get('scale', 1)
        self.use_flip = opt.get('use_flip', True) and self.phase == 'train'
        self.use_rot = opt.get('use_rot', True) and self.phase == 'train'

        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.file_client = None

        gt_folder = Path(opt['dataroot_gt'])
        lq_folder = Path(opt['dataroot_lq'])
        if not gt_folder.is_dir():
            raise FileNotFoundError(f'dataroot_gt not found: {gt_folder}')
        if not lq_folder.is_dir():
            raise FileNotFoundError(f'dataroot_lq not found: {lq_folder}')

        self.gt_paths = sorted(p for p in gt_folder.iterdir()
                               if p.suffix.lower() in IMG_EXTENSIONS)
        if len(self.gt_paths) == 0:
            raise ValueError(f'no images found in {gt_folder}')

        # match blurry frames by file name; fall back to positional pairing
        self.lq_paths = []
        for gt_path in self.gt_paths:
            lq_path = lq_folder / gt_path.name
            self.lq_paths.append(lq_path if lq_path.exists() else lq_folder / gt_path.stem)

        logger = get_root_logger()
        logger.info(f'[{self.__class__.__name__}] phase={self.phase} '
                    f'{len(self.gt_paths)} pairs from {gt_folder}')

    def __len__(self):
        return len(self.gt_paths)

    def _read(self, path):
        if self.file_client is None:
            self.file_client = FileClient(**self.io_backend_opt)
        img_bytes = self.file_client.get(str(path))
        img = imfrombytes(img_bytes, float32=True)
        return img

    def __getitem__(self, index):
        gt_path, lq_path = self.gt_paths[index], self.lq_paths[index]

        img_gt = self._read(gt_path)
        img_lq = self._read(lq_path)

        # paired random crop, so that blur and sharp stay spatially aligned
        if self.gt_size > 0:
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, self.scale, str(gt_path)
            )

        # matched geometric augmentation for both branches
        if self.use_flip or self.use_rot:
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.use_flip, self.use_rot, str(gt_path)
            )

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
        }
