import torch
from torchvision.transforms import v2
import torch.nn as nn
from PIL import Image
import os

# Import image and convert to tensor
toTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
img = toTensor(Image.open("./images/image1.jpg").convert("RGB")).unsqueeze(0)

# Define parameters
channels = img.shape[1]
kernel_size = 3
stride = 1
padding = kernel_size // 2
bias = False

# Define sharpening kernel
kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)

# Repeat the kernel for each channel
kernel = kernel.repeat(channels, 1, 1, 1)

# Create sharpening layer
sharpening = nn.Conv2d(
    in_channels=channels,
    out_channels=channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    bias=bias,
    groups=channels,
)

# Move tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = img.to(device)
kernel = kernel.to(device)
sharpening = sharpening.to(device)

# Apply sharpening
sharpening.weight = nn.Parameter(kernel, requires_grad=False)
sharpening_img = sharpening(img)

# Define blending factor (0 = original image, 1 = sharpened image, 0.5 = average of both, higher = more sharpened)
blend_factor = 1

# Blend sharpened image with original image
sharpening_img = torch.add(blend_factor * sharpening_img, (1 - blend_factor) * img)

# Clamp values between 0 and 1
sharpening_img = torch.clamp(sharpening_img, 0, 1)

# Create directory if it doesn't exist
if not os.path.exists("./images/sharpening_output"):
    os.makedirs("./images/sharpening_output")

# Save image
sharpening_img = v2.ToPILImage()(sharpening_img.squeeze(0)).convert("RGB")
sharpening_img.save("./images/sharpening_output/sharpening.jpg")
