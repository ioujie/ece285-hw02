"""
ECE 285 Homework 2: WikiArt Painting Generation with Self-Attention SNGAN-ResNet.

Consolidated standalone Python script converted from HW02.ipynb, optimized for
NVIDIA RTX 5070 Ti (BF16 AMP, channels_last, memmap preprocessing, and GPU
antialias Inception-v3 evaluation compatible with PyTorch-Ignite FID / IS).
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageOps
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

# Allow loading ultra-high-resolution paintings without triggering PIL DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# Optional PyTorch-Ignite backend for direct comparison with the reference blog
try:
    from ignite.engine import Engine as IgniteEngine
    from ignite.metrics import FID as IgniteFID, InceptionScore as IgniteIS

    HAS_IGNITE = True
except ImportError:
    HAS_IGNITE = False


# ==============================================================================
# 1. Global Configuration & Reproducibility
# ==============================================================================


def build_default_config() -> Dict:
    return {
        # Directory paths
        "data_dir": "./data",
        "cache_dir": "./cache",
        "output_dir": "./outputs",
        # Resolution and preprocessing (128x128 balances detail and runtime; 64x64 is faster)
        "image_size": 128,
        "num_preprocess_workers": min(16, os.cpu_count() or 8),
        # Architecture (Self-Attention SNGAN-ResNet)
        "z_dim": 128,
        "use_self_attention": True,
        "g_channels_128": [384, 384, 256, 128, 64, 32],  # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        "d_channels_128": [3, 32, 64, 128, 256, 384],    # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        "g_channels_64": [512, 256, 128, 64, 32],        # 4 -> 8 -> 16 -> 32 -> 64
        "d_channels_64": [3, 64, 128, 256, 512],         # 64 -> 32 -> 16 -> 8 -> 4
        # Optimization (TTUR + SNGAN Adam + BF16 AMP + DiffAugment + EMA)
        "batch_size": 128,
        "epochs": 25,
        "lr_G": 1e-4,
        "lr_D": 2e-4,
        "betas": [0.0, 0.999],
        "ema_decay": 0.999,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "use_diffaug": True,
        "diffaug_policy": "color,translation",
        # Evaluation and checkpointing
        "seed": 42,
        "eval_every": 5,
        "eval_num_samples": 5000,
        "final_eval_num_samples": 10000,
        "eval_batch_size": 64,
        "eval_backend": "auto",  # 'auto' | 'pytorch' | 'ignite'
        "num_sample_images": 64,
        "resume": True,
    }


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_environment(config: Dict) -> Tuple[torch.device, str, torch.dtype]:
    seed_everything(config["seed"])
    print(f"Python Executable : {sys.executable}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not detected. Please run inside an active CUDA environment.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU Device        : {gpu_name} ({gpu_vram_gb:.2f} GB VRAM)")

    # Enable cuDNN autotuner and TF32 matmul acceleration for Blackwell / Ampere+ GPUs
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    for sub in ["", "checkpoints", "samples", "figures"]:
        Path(config["output_dir"], sub).mkdir(parents=True, exist_ok=True)
    Path(config["cache_dir"]).mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16
    return device, gpu_name, amp_dtype


# ==============================================================================
# 2. WikiArt Dataset EDA & Fast Multithreaded Memmap Caching
# ==============================================================================


def scan_wikiart_dataset(config: Dict) -> Tuple[List[Tuple[str, int]], List[str]]:
    data_root = Path(config["data_dir"])
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root.resolve()}")

    style_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    class_names = [d.name for d in style_dirs]
    valid_exts = {".jpg", ".jpeg", ".png"}

    style_counts = {}
    all_image_records = []
    for class_idx, d in enumerate(style_dirs):
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
        style_counts[d.name] = len(files)
        for p in files:
            all_image_records.append((str(p), class_idx))

    print(f"Art styles count  : {len(class_names)}")
    print(f"Total paintings   : {len(all_image_records):,}")

    classes_csv = data_root / "classes.csv"
    if classes_csv.exists():
        meta_csv = pd.read_csv(classes_csv)
        print(
            f"Original width (px): min={meta_csv['width'].min()}, "
            f"median={meta_csv['width'].median():.0f}, max={meta_csv['width'].max()}"
        )
        print(
            f"Original height(px): min={meta_csv['height'].min()}, "
            f"median={meta_csv['height'].median():.0f}, max={meta_csv['height'].max()}"
        )

    # Save style distribution bar chart
    sorted_items = sorted(style_counts.items(), key=lambda kv: kv[1], reverse=True)
    names, counts = zip(*sorted_items)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.barh(names[::-1], counts[::-1], color="steelblue", alpha=0.9)
    ax.set_title(f"WikiArt Dataset Distribution across {len(class_names)} Art Styles (Total = {len(all_image_records):,})")
    ax.set_xlabel("Number of Paintings")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.savefig(Path(config["output_dir"], "figures", "wikiart_style_distribution.png"), dpi=200)
    plt.close(fig)

    return all_image_records, class_names


def resize_and_center_crop(img: Image.Image, target_size: int) -> np.ndarray:
    """Aspect-ratio-preserving short-edge resize followed by center crop."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Empty image dimensions")
    if w < h:
        new_w = target_size
        new_h = max(target_size, int(round(h * target_size / w)))
    else:
        new_h = target_size
        new_w = max(target_size, int(round(w * target_size / h)))
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    return np.asarray(img, dtype=np.uint8)


def _process_one_image(args):
    idx, path_str, label, target_size = args
    try:
        with Image.open(path_str) as im:
            arr = resize_and_center_crop(im, target_size)
        return idx, arr, label, None
    except Exception as exc:
        return idx, None, label, f"{path_str}: {exc}"


def build_or_load_cache(
    records: List[Tuple[str, int]],
    class_names: List[str],
    config: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess all high-res paintings once in parallel and stream directly to a
    uint8 numpy.memmap file to avoid per-epoch CPU JPEG decoding bottlenecks and
    Windows DataLoader multiprocessing spawn overhead.
    """
    img_size = config["image_size"]
    memmap_path = Path(config["cache_dir"], f"wikiart_{img_size}x{img_size}_rgb.npy")
    meta_path = Path(config["cache_dir"], f"wikiart_{img_size}x{img_size}_meta.json")

    if memmap_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("image_size") == img_size and meta.get("complete", False):
            arr = np.load(memmap_path, mmap_mode="r")
            print(f"[Cache Hit] Loaded {memmap_path} | shape={arr.shape}, dtype={arr.dtype}")
            return arr, meta

    n_total = len(records)
    tmp_path = memmap_path.with_suffix(".tmp.npy")
    if tmp_path.exists():
        tmp_path.unlink()

    num_workers = config["num_preprocess_workers"]
    print(f"[Cache Build] Preprocessing {n_total:,} images into {img_size}x{img_size} memmap (workers={num_workers})...")
    t0 = time.time()
    mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(n_total, img_size, img_size, 3))

    tasks = [(i, p, lbl, img_size) for i, (p, lbl) in enumerate(records)]
    valid_count = 0
    labels_list = []
    errors = []

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for _, arr, lbl, err in tqdm(pool.map(_process_one_image, tasks, chunksize=32), total=n_total, desc="Preprocessing"):
            if err is not None or arr is None:
                errors.append(err)
                continue
            mm[valid_count] = arr
            labels_list.append(int(lbl))
            valid_count += 1
            if valid_count % 4000 == 0:
                mm.flush()
    mm.flush()
    del mm

    if errors:
        print(f"[Warning] Skipped {len(errors)} corrupted images. Example: {errors[:3]}")

    if valid_count == n_total:
        tmp_path.replace(memmap_path)
    else:
        src_mm = np.load(tmp_path, mmap_mode="r")
        dst_mm = np.lib.format.open_memmap(
            memmap_path, mode="w+", dtype=np.uint8, shape=(valid_count, img_size, img_size, 3)
        )
        for start in range(0, valid_count, 4096):
            end = min(start + 4096, valid_count)
            dst_mm[start:end] = src_mm[start:end]
        dst_mm.flush()
        del src_mm, dst_mm
        tmp_path.unlink()

    # Sample uniformly across all 27 styles so FID real reference is unbiased
    rng = np.random.default_rng(config["seed"])
    eval_pool_n = min(config["final_eval_num_samples"], valid_count)
    eval_indices = rng.choice(valid_count, size=eval_pool_n, replace=False).tolist()

    meta = {
        "complete": True,
        "image_size": img_size,
        "num_images": valid_count,
        "class_names": class_names,
        "labels": labels_list,
        "eval_indices": eval_indices,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    size_gb = memmap_path.stat().st_size / (1024 ** 3)
    print(f"[Cache Done] Valid images={valid_count:,}, size={size_gb:.2f} GB, time={time.time() - t0:.1f}s")
    return np.load(memmap_path, mmap_mode="r"), meta


class MemmapBatchLoader:
    """Zero-worker memmap batch loader performing GPU-side normalization and random horizontal flip."""

    def __init__(
        self,
        memmap_arr: np.ndarray,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        drop_last: bool = True,
        random_flip: bool = True,
        indices: Optional[np.ndarray] = None,
    ):
        self.arr = memmap_arr
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_flip = random_flip
        self.base_indices = np.arange(len(memmap_arr)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        n = len(self.base_indices)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        order = self.base_indices.copy()
        if self.shuffle:
            np.random.shuffle(order)
        n = len(order)
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            batch_np = np.ascontiguousarray(self.arr[batch_idx])  # (B, H, W, 3) uint8
            x = torch.from_numpy(batch_np).to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2).float().mul_(1.0 / 127.5).sub_(1.0)  # Scale to [-1, 1]
            if self.random_flip:
                flip_mask = torch.rand(x.size(0), 1, 1, 1, device=self.device) < 0.5
                x = torch.where(flip_mask, x.flip(-1), x)
            yield x.contiguous(memory_format=torch.channels_last)


# ==============================================================================
# 3. Model Architecture: Self-Attention SNGAN-ResNet
# ==============================================================================


class SelfAttention(nn.Module):
    """SAGAN-style non-local self-attention using PyTorch scaled_dot_product_attention."""

    def __init__(self, channels: int, use_sn: bool = True):
        super().__init__()
        self.ch_qk = max(channels // 8, 16)
        self.ch_v = channels // 2
        qkv_conv = nn.Conv2d(channels, self.ch_qk * 2 + self.ch_v, kernel_size=1, bias=False)
        out_conv = nn.Conv2d(self.ch_v, channels, kernel_size=1, bias=False)
        self.qkv = spectral_norm(qkv_conv) if use_sn else qkv_conv
        self.out = spectral_norm(out_conv) if use_sn else out_conv
        self.gamma = nn.Parameter(torch.zeros(1))  # Starts as identity for stable early training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q, k, v = torch.split(self.qkv(x), [self.ch_qk, self.ch_qk, self.ch_v], dim=1)
        q = q.flatten(2).transpose(1, 2).unsqueeze(1)
        k = k.flatten(2).transpose(1, 2).unsqueeze(1)
        v = v.flatten(2).transpose(1, 2).unsqueeze(1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.squeeze(1).transpose(1, 2).reshape(b, self.ch_v, h, w)
        return x + self.gamma * self.out(attn)


class ResBlockUp(nn.Module):
    """Generator residual upsampling block using nearest interpolation + conv3x3 to prevent checkerboard artifacts."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(F.silu(self.bn1(x)), scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.silu(self.bn2(h)))
        s = self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))
        return h + s


class ResBlockDown(nn.Module):
    """Discriminator spectral-normalized residual downsampling block."""

    def __init__(self, in_c: int, out_c: int, first: bool = False):
        super().__init__()
        self.first = first  # Skip pre-activation on raw [-1, 1] RGB inputs to retain dark tones
        self.conv1 = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.first else F.leaky_relu(x, 0.2, inplace=False)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2, inplace=False)
        h = F.avg_pool2d(self.conv2(h), kernel_size=2)
        s = self.skip(F.avg_pool2d(x, kernel_size=2))
        return h + s


class Generator(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        image_size = config["image_size"]
        if image_size not in (64, 128):
            raise ValueError(f"Only image_size 64 or 128 is supported, got {image_size}")
        ch = config["g_channels_128"] if image_size == 128 else config["g_channels_64"]
        self.c0 = ch[0]
        self.fc = nn.Linear(config["z_dim"], self.c0 * 4 * 4)

        layers = []
        cur_res = 4
        for i in range(len(ch) - 1):
            layers.append(ResBlockUp(ch[i], ch[i + 1]))
            cur_res *= 2
            if config["use_self_attention"] and cur_res == 32:
                layers.append(SelfAttention(ch[i + 1], use_sn=True))
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(ch[-1])
        self.out = nn.Conv2d(ch[-1], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), -1)
        h = self.fc(z).reshape(z.size(0), self.c0, 4, 4)
        h = self.blocks(h)
        return torch.tanh(self.out(F.silu(self.bn(h))))


class Discriminator(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        image_size = config["image_size"]
        if image_size not in (64, 128):
            raise ValueError(f"Only image_size 64 or 128 is supported, got {image_size}")
        ch = config["d_channels_128"] if image_size == 128 else config["d_channels_64"]

        layers = []
        cur_res = image_size
        for i in range(len(ch) - 1):
            layers.append(ResBlockDown(ch[i], ch[i + 1], first=(i == 0)))
            cur_res //= 2
            if config["use_self_attention"] and cur_res == 32:
                layers.append(SelfAttention(ch[i + 1], use_sn=True))
        self.blocks = nn.Sequential(*layers)
        self.fc = spectral_norm(nn.Linear(ch[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.blocks(x), 0.2, inplace=False)
        h = h.sum(dim=(2, 3))  # Global Sum Pooling
        return self.fc(h).squeeze(1)


def init_gan_weights(module: nn.Module) -> None:
    """Orthogonal initialization supporting both spectral_norm parametrizations and channels_last layouts."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            w = module.parametrizations.weight.original
        else:
            w = module.weight
        if w.is_contiguous():
            nn.init.orthogonal_(w)
        else:
            tmp = torch.empty_like(w, memory_format=torch.contiguous_format)
            nn.init.orthogonal_(tmp)
            with torch.no_grad():
                w.copy_(tmp)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update EMA weights and synchronize BatchNorm running buffers."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.lerp_(p.data, 1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 4. DiffAugment & Hinge Loss
# ==============================================================================


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    shift = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + shift * 0.5


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 1.5 + 0.25
    return (x - x_mean) * scale + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * scale + x_mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    b, _, h, w = x.shape
    shift_x, shift_y = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=(b, 1, 1), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=(b, 1, 1), device=x.device)
    grid_b, grid_x, grid_y = torch.meshgrid(
        torch.arange(b, dtype=torch.long, device=x.device),
        torch.arange(h, dtype=torch.long, device=x.device),
        torch.arange(w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + tx + 1, 0, h + 1)
    grid_y = torch.clamp(grid_y + ty + 1, 0, w + 1)
    x_pad = F.pad(x, (1, 1, 1, 1))
    out = x_pad.permute(0, 2, 3, 1).contiguous()[grid_b, grid_x, grid_y].permute(0, 3, 1, 2)
    return out.contiguous(memory_format=torch.channels_last)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
}


def diff_augment(x: torch.Tensor, policy: str = "color,translation") -> torch.Tensor:
    if not policy:
        return x
    for p in policy.split(","):
        for fn in AUGMENT_FNS.get(p.strip(), []):
            x = fn(x)
    return x


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


# ==============================================================================
# 5. PyTorch-Ignite Compatible FID & Inception Score Evaluator
# ==============================================================================


def preprocess_for_inception(imgs_neg1_1: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] outputs to [0, 1], antialias bilinear resize to 299x299 on GPU, and apply ImageNet norm."""
    x = (imgs_neg1_1.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    if x.shape[-2:] != (299, 299):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
    return (x - mean) / std


class InceptionEvaluator:
    """Zero-extra-dependency GPU evaluator matching PyTorch-Ignite FID and InceptionScore."""

    def __init__(self, device: torch.device, batch_size: int = 64, use_amp: bool = True):
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp

        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, transform_input=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._feat_buf = []
        self.model.avgpool.register_forward_hook(self._hook_fn)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _hook_fn(self, module, inputs, output) -> None:
        self._feat_buf.append(output.flatten(1))

    @torch.no_grad()
    def extract_from_memmap(self, memmap_arr: np.ndarray, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        feats, probs = [], []
        idx_arr = np.asarray(indices, dtype=np.int64)
        for start in tqdm(range(0, len(idx_arr), self.batch_size), desc="Inception(Real)", leave=False):
            sub = idx_arr[start:start + self.batch_size]
            batch_np = np.ascontiguousarray(memmap_arr[sub])
            x = torch.from_numpy(batch_np).to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            inp = preprocess_for_inception(x, self.mean, self.std)
            self._feat_buf.clear()
            logits = self.model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            feats.append(self._feat_buf.pop().cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0)

    @torch.no_grad()
    def extract_from_generator(
        self, generator: nn.Module, num_samples: int, z_dim: int, amp_dtype: torch.dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
        generator.eval()
        feats, probs = [], []
        for start in tqdm(range(0, num_samples, self.batch_size), desc="Inception(Fake)", leave=False):
            cur_b = min(self.batch_size, num_samples - start)
            z = torch.randn(cur_b, z_dim, device=self.device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self.use_amp):
                fake = generator(z)
            inp = preprocess_for_inception(fake, self.mean, self.std)
            self._feat_buf.clear()
            logits = self.model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            feats.append(self._feat_buf.pop().cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(probs, axis=0)

    @staticmethod
    def compute_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(features, axis=0), np.cov(features, rowvar=False)

    @staticmethod
    def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
        diff = np.atleast_1d(mu1) - np.atleast_1d(mu2)
        covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print(f"[FID Warning] Imaginary component {m:.3e} truncated.")
            covmean = covmean.real
        fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
        return max(fid, 0.0)

    @staticmethod
    def compute_inception_score(probs: np.ndarray, splits: int = 10) -> Tuple[float, float]:
        n = probs.shape[0]
        split_scores = []
        for k in range(splits):
            part = probs[k * (n // splits):(k + 1) * (n // splits), :]
            if len(part) == 0:
                continue
            py = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
            split_scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
        return float(np.mean(split_scores)), float(np.std(split_scores))


def evaluate_with_ignite(
    generator: nn.Module,
    memmap_arr: np.ndarray,
    real_indices: List[int],
    z_dim: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Optional native PyTorch-Ignite Engine evaluation matching the reference blog."""
    if not HAS_IGNITE:
        raise ImportError("pytorch-ignite is not installed in the current environment.")
    generator.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    real_idx_arr = np.asarray(real_indices, dtype=np.int64)
    batches = [real_idx_arr[i:i + batch_size] for i in range(0, len(real_idx_arr), batch_size)]

    def eval_step(engine, sub_idx):
        with torch.no_grad():
            z = torch.randn(len(sub_idx), z_dim, device=device)
            fake = generator(z)
            real_np = np.ascontiguousarray(memmap_arr[sub_idx])
            real = torch.from_numpy(real_np).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            return preprocess_for_inception(fake, mean, std), preprocess_for_inception(real, mean, std)

    evaluator = IgniteEngine(eval_step)
    IgniteFID(device=device).attach(evaluator, "fid")
    IgniteIS(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")
    state = evaluator.run(batches, max_epochs=1)
    return float(state.metrics["fid"]), float(state.metrics["is"]), 0.0


def get_or_cache_real_stats(
    evaluator_core: InceptionEvaluator,
    images_mm: np.ndarray,
    dataset_meta: Dict,
    config: Dict,
    num_samples: int,
):
    img_size = config["image_size"]
    stats_path = Path(config["cache_dir"], f"real_stats_{img_size}_{num_samples}.npz")
    if stats_path.exists():
        data = np.load(stats_path)
        print(f"[Real Stats Hit] {stats_path}")
        return data["mu"], data["sigma"], float(data["real_is_mean"]), float(data["real_is_std"])

    indices = dataset_meta["eval_indices"][:num_samples]
    feats, probs = evaluator_core.extract_from_memmap(images_mm, indices)
    mu, sigma = evaluator_core.compute_stats(feats)
    is_mean, is_std = evaluator_core.compute_inception_score(probs, splits=10)
    np.savez(stats_path, mu=mu, sigma=sigma, real_is_mean=is_mean, real_is_std=is_std)
    print(f"[Real Stats Saved] {stats_path} | Real WikiArt IS = {is_mean:.3f} +/- {is_std:.3f}")
    return mu, sigma, is_mean, is_std


def evaluate_gan(
    generator: nn.Module,
    evaluator_core: InceptionEvaluator,
    real_stats: Tuple[np.ndarray, np.ndarray, float, float],
    num_samples: int,
    config: Dict,
    amp_dtype: torch.dtype,
    device: torch.device,
    images_mm: Optional[np.ndarray] = None,
    dataset_meta: Optional[Dict] = None,
    use_final_stats: bool = False,
) -> Tuple[float, float, float]:
    backend = config["eval_backend"]
    if backend == "ignite" or (backend == "auto" and HAS_IGNITE and use_final_stats):
        assert images_mm is not None and dataset_meta is not None
        return evaluate_with_ignite(
            generator,
            images_mm,
            dataset_meta["eval_indices"][:num_samples],
            config["z_dim"],
            config["eval_batch_size"],
            device,
        )

    mu_r, sigma_r, _, _ = real_stats
    feats_g, probs_g = evaluator_core.extract_from_generator(generator, num_samples, config["z_dim"], amp_dtype)
    mu_g, sigma_g = evaluator_core.compute_stats(feats_g)
    fid = evaluator_core.compute_fid(mu_r, sigma_r, mu_g, sigma_g)
    is_mean, is_std = evaluator_core.compute_inception_score(probs_g, splits=10)
    torch.cuda.empty_cache()
    return fid, is_mean, is_std


# ==============================================================================
# 6. Training Loop & Visualization Utilities
# ==============================================================================


@torch.no_grad()
def save_sample_grid(
    generator: nn.Module, z: torch.Tensor, save_path: Path, amp_dtype: torch.dtype, use_amp: bool
) -> None:
    generator.eval()
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        imgs = generator(z)
    imgs = (imgs.float().clamp(-1.0, 1.0) + 1.0) * 0.5
    vutils.save_image(imgs, save_path, nrow=8, padding=2)


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation preserving norm on the Gaussian latent hypersphere."""
    low_n = low / torch.norm(low, dim=-1, keepdim=True).clamp_min(1e-8)
    high_n = high / torch.norm(high, dim=-1, keepdim=True).clamp_min(1e-8)
    omega = torch.acos(torch.clamp((low_n * high_n).sum(dim=-1, keepdim=True), -0.9999, 0.9999))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return torch.where(so.abs() < 1e-6, (1.0 - val) * low + val * high, res)


def moving_average(arr, window: int = 100) -> np.ndarray:
    arr = np.asarray(arr)
    if len(arr) < window:
        window = max(1, len(arr) // 5)
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def train_gan(
    netG: nn.Module,
    netD: nn.Module,
    netG_ema: nn.Module,
    train_loader: MemmapBatchLoader,
    evaluator_core: InceptionEvaluator,
    real_stats_eval: Tuple[np.ndarray, np.ndarray, float, float],
    config: Dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    images_mm: np.ndarray,
    dataset_meta: Dict,
) -> Dict:
    optG = torch.optim.Adam(netG.parameters(), lr=config["lr_G"], betas=tuple(config["betas"]))
    optD = torch.optim.Adam(netD.parameters(), lr=config["lr_D"], betas=tuple(config["betas"]))

    ckpt_dir = Path(config["output_dir"], "checkpoints")
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    history_path = Path(config["output_dir"], "history.json")

    fixed_noise = torch.randn(config["num_sample_images"], config["z_dim"], device=device)

    history = {
        "iter_loss_G": [],
        "iter_loss_D": [],
        "iter_D_real": [],
        "iter_D_fake": [],
        "epoch_loss_G": [],
        "epoch_loss_D": [],
        "eval_epochs": [],
        "fid_scores": [],
        "is_means": [],
        "is_stds": [],
        "best_fid": float("inf"),
        "best_epoch": 0,
        "peak_vram_gb": 0.0,
    }

    start_epoch = 1
    global_step = 0

    if config["resume"] and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        netG_ema.load_state_dict(ckpt["netG_ema"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        fixed_noise = ckpt["fixed_noise"].to(device)
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"[Resume] Resumed from epoch {ckpt['epoch']} (step={global_step}, best_fid={history['best_fid']:.2f})")

    print(f"Training: epochs {start_epoch} -> {config['epochs']} | AMP={config['use_amp']} ({config['amp_dtype']})")
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, config["epochs"] + 1):
        netG.train()
        netD.train()
        ep_t0 = time.time()
        ep_g_losses, ep_d_losses = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config['epochs']:02d}")
        for real in pbar:
            global_step += 1
            b_size = real.size(0)
            z = torch.randn(b_size, config["z_dim"], device=device)

            # (1) Discriminator update
            optD.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config["use_amp"]):
                fake = netG(z)
                real_aug = diff_augment(real, config["diffaug_policy"]) if config["use_diffaug"] else real
                fake_aug_d = (
                    diff_augment(fake.detach(), config["diffaug_policy"]) if config["use_diffaug"] else fake.detach()
                )
                d_real = netD(real_aug)
                d_fake = netD(fake_aug_d)
                loss_d = d_hinge_loss(d_real, d_fake)
            loss_d.backward()
            optD.step()

            # (2) Generator update (reusing `fake` computation graph saves ~33% forward time)
            for p in netD.parameters():
                p.requires_grad_(False)
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config["use_amp"]):
                fake_aug_g = diff_augment(fake, config["diffaug_policy"]) if config["use_diffaug"] else fake
                loss_g = g_hinge_loss(netD(fake_aug_g))
            loss_g.backward()
            optG.step()
            for p in netD.parameters():
                p.requires_grad_(True)

            # (3) EMA update with warmup
            ema_decay = min(config["ema_decay"], (1.0 + global_step) / (10.0 + global_step))
            update_ema(netG_ema, netG, decay=ema_decay)

            lg, ld = float(loss_g.item()), float(loss_d.item())
            dr, df_val = float(d_real.mean().item()), float(d_fake.mean().item())
            ep_g_losses.append(lg)
            ep_d_losses.append(ld)
            history["iter_loss_G"].append(lg)
            history["iter_loss_D"].append(ld)
            history["iter_D_real"].append(dr)
            history["iter_D_fake"].append(df_val)

            if global_step % 20 == 0:
                pbar.set_postfix(loss_D=f"{ld:.3f}", loss_G=f"{lg:.3f}", D_x=f"{dr:.2f}", D_gz=f"{df_val:.2f}")

        ep_time = time.time() - ep_t0
        mean_lg, mean_ld = float(np.mean(ep_g_losses)), float(np.mean(ep_d_losses))
        history["epoch_loss_G"].append(mean_lg)
        history["epoch_loss_D"].append(mean_ld)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        history["peak_vram_gb"] = max(float(history.get("peak_vram_gb", 0.0)), float(peak_vram_gb))

        log_msg = (
            f"Epoch [{epoch:02d}/{config['epochs']:02d}] "
            f"Time: {ep_time:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GB | "
            f"Loss_D: {mean_ld:.4f} | Loss_G: {mean_lg:.4f}"
        )

        should_eval = (epoch == 1) or (epoch % config["eval_every"] == 0) or (epoch == config["epochs"])
        if should_eval:
            save_sample_grid(
                netG_ema,
                fixed_noise,
                Path(config["output_dir"], "samples", f"epoch_{epoch:03d}.png"),
                amp_dtype,
                config["use_amp"],
            )
            fid_val, is_mean, is_std = evaluate_gan(
                netG_ema,
                evaluator_core,
                real_stats_eval,
                config["eval_num_samples"],
                config,
                amp_dtype,
                device,
                images_mm=images_mm,
                dataset_meta=dataset_meta,
                use_final_stats=False,
            )
            history["eval_epochs"].append(epoch)
            history["fid_scores"].append(fid_val)
            history["is_means"].append(is_mean)
            history["is_stds"].append(is_std)
            log_msg += f" | FID: {fid_val:.2f} | IS: {is_mean:.3f} +/- {is_std:.3f}"

            if fid_val < history["best_fid"]:
                history["best_fid"] = fid_val
                history["best_epoch"] = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "netG_ema": netG_ema.state_dict(),
                        "netG": netG.state_dict(),
                        "fid": fid_val,
                        "is_mean": is_mean,
                        "is_std": is_std,
                        "config": config,
                    },
                    best_ckpt_path,
                )
                log_msg += "  [New Best!]"

        print(log_msg)

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "netG_ema": netG_ema.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "fixed_noise": fixed_noise.cpu(),
                "history": history,
            },
            last_ckpt_path,
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history


def export_figures(
    config: Dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    netG_ema: nn.Module,
    images_mm: Optional[np.ndarray] = None,
    dataset_meta: Optional[Dict] = None,
) -> None:
    out_dir = Path(config["output_dir"])
    fig_dir = out_dir / "figures"
    history_path = out_dir / "history.json"
    best_ckpt_path = out_dir / "checkpoints" / "best_checkpoint.pt"
    summary_path = out_dir / "metrics_summary.json"
    img_size = config["image_size"]

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

        # Figure 1: Training loss curves & discriminator equilibrium
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
        g_raw = np.asarray(hist["iter_loss_G"])
        d_raw = np.asarray(hist["iter_loss_D"])
        w_ma = min(200, max(10, len(g_raw) // 30))
        g_ma = moving_average(g_raw, w_ma)
        d_ma = moving_average(d_raw, w_ma)

        axes[0].plot(g_raw, color="tab:orange", alpha=0.18)
        axes[0].plot(d_raw, color="tab:blue", alpha=0.18)
        axes[0].plot(np.arange(len(g_ma)) + (len(g_raw) - len(g_ma)), g_ma, color="tab:orange", lw=2.0, label="Generator Loss (MA)")
        axes[0].plot(np.arange(len(d_ma)) + (len(d_raw) - len(d_ma)), d_ma, color="tab:blue", lw=2.0, label="Discriminator Hinge Loss (MA)")
        axes[0].set_title("GAN Training Loss vs Iterations")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend()

        dr_ma = moving_average(hist["iter_D_real"], w_ma)
        df_ma = moving_average(hist["iter_D_fake"], w_ma)
        axes[1].plot(np.arange(len(dr_ma)) + (len(g_raw) - len(dr_ma)), dr_ma, color="tab:green", lw=2.0, label="Mean D(real)")
        axes[1].plot(np.arange(len(df_ma)) + (len(g_raw) - len(df_ma)), df_ma, color="tab:red", lw=2.0, label="Mean D(G(z))")
        axes[1].axhline(0.0, color="gray", linestyle=":", lw=1.0)
        axes[1].set_title("Discriminator Logit Equilibrium")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean Discriminator Score")
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend()
        fig.savefig(fig_dir / "loss_curves.png", dpi=300)
        plt.close(fig)

        # Figure 2: FID & IS progression
        epochs_ev = np.asarray(hist["eval_epochs"])
        fids = np.asarray(hist["fid_scores"])
        is_m = np.asarray(hist["is_means"])
        is_s = np.asarray(hist["is_stds"])

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        c_fid, c_is = "tab:blue", "tab:red"
        l1 = ax1.plot(epochs_ev, fids, "o-", color=c_fid, lw=2.2, label="FID (lower is better)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("FID", color=c_fid)
        ax1.tick_params(axis="y", labelcolor=c_fid)
        ax1.grid(True, linestyle="--", alpha=0.35)

        best_idx = int(np.argmin(fids))
        ax1.annotate(
            f"Best FID: {fids[best_idx]:.2f}\n(Ep {epochs_ev[best_idx]})",
            xy=(epochs_ev[best_idx], fids[best_idx]),
            xytext=(epochs_ev[best_idx], fids[best_idx] + (fids.max() - fids.min()) * 0.12),
            arrowprops=dict(arrowstyle="->", color=c_fid),
            fontsize=9,
            color=c_fid,
            fontweight="bold",
        )

        ax2 = ax1.twinx()
        l2 = ax2.plot(epochs_ev, is_m, "s--", color=c_is, lw=2.2, label="Inception Score (higher is better)")
        ax2.fill_between(epochs_ev, is_m - is_s, is_m + is_s, color=c_is, alpha=0.15)
        ax2.set_ylabel("Inception Score", color=c_is)
        ax2.tick_params(axis="y", labelcolor=c_is)

        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
        plt.title("FID and Inception Score Progression Across Training Epochs")
        fig.savefig(fig_dir / "fid_is_curves.png", dpi=300)
        plt.close(fig)

    # Load best EMA checkpoint for visual generation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    netG_ema.load_state_dict(best_ckpt["netG_ema"])
    netG_ema.eval()

    with torch.no_grad():
        z_show = torch.randn(64, config["z_dim"], device=device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config["use_amp"]):
            fake_64 = (netG_ema(z_show).float().clamp(-1.0, 1.0) + 1.0) * 0.5
        fake_64 = fake_64.cpu()

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            _sum = json.load(f)
        vis_fid, vis_is = _sum["final_fid"], _sum["final_is_mean"]
    else:
        vis_fid, vis_is = float(best_ckpt["fid"]), float(best_ckpt["is_mean"])

    # Figure 3: Real vs Fake comparison (when dataset memmap is available)
    if images_mm is not None and dataset_meta is not None:
        real_idx = dataset_meta["eval_indices"][:64]
        real_64 = torch.from_numpy(np.ascontiguousarray(images_mm[real_idx])).permute(0, 3, 1, 2).float() / 255.0

        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
        axes[0].axis("off")
        axes[0].set_title(f"Real WikiArt Paintings ({img_size}x{img_size})", fontsize=13)
        axes[0].imshow(vutils.make_grid(real_64, nrow=8, padding=2).permute(1, 2, 0).numpy())
        axes[1].axis("off")
        axes[1].set_title(f"Generated Paintings (Best EMA @ Ep {best_ckpt['epoch']}, FID={vis_fid:.2f}, IS={vis_is:.2f})", fontsize=13)
        axes[1].imshow(vutils.make_grid(fake_64, nrow=8, padding=2).permute(1, 2, 0).numpy())
        fig.savefig(fig_dir / "real_vs_fake.png", dpi=300)
        plt.close(fig)

    # Figure 4: Generated 6x6 gallery
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    ax.set_title(f"Generated WikiArt Gallery ({img_size}x{img_size})", fontsize=13)
    ax.imshow(vutils.make_grid(fake_64[:36], nrow=6, padding=2).permute(1, 2, 0).numpy())
    fig.savefig(fig_dir / "generated_gallery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: Latent space SLERP interpolation (8x8)
    with torch.no_grad():
        z_a = torch.randn(8, config["z_dim"], device=device)
        z_b = torch.randn(8, config["z_dim"], device=device)
        interp_zs = [
            slerp(float(alpha), z_a[i], z_b[i])
            for i in range(8)
            for alpha in np.linspace(0.0, 1.0, 8)
        ]
        interp_z = torch.stack(interp_zs, dim=0)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=config["use_amp"]):
            interp_imgs = (netG_ema(interp_z).float().clamp(-1.0, 1.0) + 1.0) * 0.5

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.axis("off")
    ax.set_title("Latent Space Spherical Linear Interpolation (SLERP)", fontsize=13)
    ax.imshow(vutils.make_grid(interp_imgs.cpu(), nrow=8, padding=2).permute(1, 2, 0).numpy())
    fig.savefig(fig_dir / "latent_interpolation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {fig_dir.resolve()}")


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="WikiArt Painting Generation with Self-Attention SNGAN-ResNet")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to WikiArt data directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to preprocessed memmap cache")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output artifacts directory")
    parser.add_argument("--image-size", type=int, choices=[64, 128], default=None, help="Target image resolution")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--eval-backend", type=str, choices=["auto", "pytorch", "ignite"], default=None)
    parser.add_argument("--no-resume", action="store_true", help="Train from scratch instead of resuming")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate best_checkpoint.pt and export figures")
    parser.add_argument("--sample-only", action="store_true", help="Generate gallery & SLERP figures from best_checkpoint.pt without dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_default_config()

    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.eval_backend is not None:
        config["eval_backend"] = args.eval_backend
    if args.no_resume:
        config["resume"] = False

    device, gpu_name, amp_dtype = setup_environment(config)

    # Build models (always initialize weights in contiguous format before channels_last conversion)
    netG = Generator(config)
    netD = Discriminator(config)
    netG.apply(init_gan_weights)
    netD.apply(init_gan_weights)
    netG = netG.to(device, memory_format=torch.channels_last)
    netD = netD.to(device, memory_format=torch.channels_last)

    netG_ema = copy.deepcopy(netG).eval()
    for p in netG_ema.parameters():
        p.requires_grad_(False)

    print(f"Generator Parameters     : {count_parameters(netG) / 1e6:.2f} M")
    print(f"Discriminator Parameters : {count_parameters(netD) / 1e6:.2f} M")

    if args.sample_only:
        export_figures(config, device, amp_dtype, netG_ema)
        return

    with open(Path(config["output_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    records, class_names = scan_wikiart_dataset(config)
    images_mm, dataset_meta = build_or_load_cache(records, class_names, config)

    evaluator_core = InceptionEvaluator(device=device, batch_size=config["eval_batch_size"], use_amp=config["use_amp"])
    real_stats_eval = get_or_cache_real_stats(evaluator_core, images_mm, dataset_meta, config, config["eval_num_samples"])
    real_stats_final = get_or_cache_real_stats(
        evaluator_core, images_mm, dataset_meta, config, config["final_eval_num_samples"]
    )

    history = {}
    if not args.eval_only:
        train_loader = MemmapBatchLoader(
            images_mm,
            batch_size=config["batch_size"],
            device=device,
            shuffle=True,
            drop_last=True,
            random_flip=True,
        )
        history = train_gan(
            netG=netG,
            netD=netD,
            netG_ema=netG_ema,
            train_loader=train_loader,
            evaluator_core=evaluator_core,
            real_stats_eval=real_stats_eval,
            config=config,
            device=device,
            amp_dtype=amp_dtype,
            images_mm=images_mm,
            dataset_meta=dataset_meta,
        )
    else:
        history_path = Path(config["output_dir"], "history.json")
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)

    # Final 10k-sample evaluation on best EMA checkpoint
    best_ckpt_path = Path(config["output_dir"], "checkpoints", "best_checkpoint.pt")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    netG_ema.load_state_dict(best_ckpt["netG_ema"])
    netG_ema.eval()

    final_fid, final_is_mean, final_is_std = evaluate_gan(
        netG_ema,
        evaluator_core,
        real_stats_final,
        config["final_eval_num_samples"],
        config,
        amp_dtype,
        device,
        images_mm=images_mm,
        dataset_meta=dataset_meta,
        use_final_stats=True,
    )

    summary = {
        "model": f"Self-Attention SNGAN-ResNet ({config['image_size']}x{config['image_size']})",
        "gpu": gpu_name,
        "dataset_images": int(dataset_meta["num_images"]),
        "best_epoch": int(best_ckpt["epoch"]),
        "eval_samples": int(config["final_eval_num_samples"]),
        "final_fid": round(final_fid, 4),
        "final_is_mean": round(final_is_mean, 4),
        "final_is_std": round(final_is_std, 4),
        "real_wikiart_is_mean": round(real_stats_final[2], 4),
        "real_wikiart_is_std": round(real_stats_final[3], 4),
        "g_params_millions": round(count_parameters(netG) / 1e6, 3),
        "d_params_millions": round(count_parameters(netD) / 1e6, 3),
        "peak_vram_gb": round(
            max(float(history.get("peak_vram_gb", 0.0)), torch.cuda.max_memory_allocated(device) / (1024 ** 3)), 3
        ),
    }

    with open(Path(config["output_dir"], "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    print("           FINAL WIKIART GAN EVALUATION SUMMARY")
    print("=" * 56)
    for k, v in summary.items():
        print(f"{k:24s}: {v}")
    print("=" * 56)

    export_figures(config, device, amp_dtype, netG_ema, images_mm=images_mm, dataset_meta=dataset_meta)


if __name__ == "__main__":
    main()
