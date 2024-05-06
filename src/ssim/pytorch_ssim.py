import torch
from torchvision.transforms import v2
from PIL import Image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),  # scale=False for [0, 255] range
    ]
)

img1 = (
    transform(Image.open("./images/image13.jpg").convert("RGB")).unsqueeze(0).to(device)
)
img2 = (
    transform(Image.open("./images/image14.jpg").convert("RGB")).unsqueeze(0).to(device)
)

# https://pypi.org/project/pytorch-msssim/
# https://github.com/VainF/pytorch-msssim
# Using the ssim and ms_ssim functions
# NOTE: The ssim and ms_ssim don't need to be moved to the GPU, as they use the same device as the input tensors (cuda in this case)
ssim_val = 1 - ssim(img1, img2, data_range=255, size_average=True)
print(f"ssim: {ssim_val.item() * 100:.2f}%")
ms_ssim_val = 1 - ms_ssim(img1, img2, data_range=255, size_average=True)
print(f"ms_ssim: {ms_ssim_val.item() * 100:.2f}%")

# Using the SSIM and MS_SSIM classes
ssim_module = SSIM(data_range=255, size_average=True, channel=3).to(device)
ssim_val = 1 - ssim_module(img1, img2)
print(f"SSIM: {ssim_val.item() * 100:.2f}%")
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3).to(device)
ms_ssim_val = 1 - ms_ssim_module(img1, img2)
print(f"MS_SSIM: {ms_ssim_val.item() * 100:.2f}%")
