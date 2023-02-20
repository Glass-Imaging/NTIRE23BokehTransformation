import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

lpips_alex = lpips.LPIPS(net="alex").cuda()


def calculate_lpips(img0: torch.Tensor, img1: torch.Tensor):
    # NOTE: LPIPS expects image normalized to [-1, 1]
    img0 = 2 * img0 - 1.0
    img1 = 2 * img1 - 1.0

    with torch.no_grad():
        distance = lpips_alex(img0, img1)
    return distance.item()


def calculate_psnr(img0, img1):
    mse = np.mean((img0.astype(np.float32) - img1.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    max_val = np.max(img0)
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img0, img1):
    val = ssim(img0, img1, channel_axis=2)
    return val
