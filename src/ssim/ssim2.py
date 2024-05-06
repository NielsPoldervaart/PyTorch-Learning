import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import Tuple
from PIL import Image


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    channel_avg: bool = True,
    padding: bool = False,
    value_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Structural Similarity Index (SSIM) and Contrast Sensitivity (CS) between two tensors.

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        kernel (torch.Tensor): Smoothing kernel.
        channel_avg (bool): Whether to average over channels.
        padding (bool): Whether to pad spatial dimensions.
        value_range (float): Value range of the inputs.
        k1 (float): Constant for SSIM calculation.
        k2 (float): Constant for SSIM calculation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: SSIM and CS tensors.
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    window = kernel.repeat(1, 1, x.dim() - 2)

    pad = kernel.shape[-1] // 2 if padding else 0

    # Mean (mu)
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)

    mu_xx = mu_x**2
    mu_yy = mu_y**2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = F.conv2d(x**2, window, padding=pad) - mu_xx
    sigma_yy = F.conv2d(y**2, window, padding=pad) - mu_yy
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    # Contrast sensitivity (CS)
    cs = (2 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    if channel_avg:
        ss, cs = ss.mean(dim=1), cs.mean(dim=1)
    else:
        ss, cs = ss.mean(dim=2), cs.mean(dim=2)

    return ss.mean(dim=-1), cs.mean(dim=-1)


transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

img1 = transform(Image.open("image1.jpg").convert("RGB")).unsqueeze(0)
img2 = transform(Image.open("image2.jpg").convert("RGB")).unsqueeze(0)

similarity = ssim(img1, img2)

print(similarity)
