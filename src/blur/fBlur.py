import torch
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
import os

toTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

img = toTensor(Image.open("./images/image1.jpg").convert("RGB")).unsqueeze(0)

# Define parameters
sigma = 5.0
kernel_size = int(6 * sigma + 1)
kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
channels = img.shape[1]
stride = 1

# Create Gaussian kernel
ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
xx, yy = torch.meshgrid(ax, ax, indexing="xy")
kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
kernel = kernel / torch.sum(kernel)
kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

# Setup Conv2d layer
gaussian_blur = F.conv2d(
    input=img, weight=kernel, stride=stride, padding=kernel_size // 2, groups=channels
)

# Create output folder if not exists
if not os.path.exists("./images/blur_output"):
    os.makedirs("./images/blur_output")

# Apply blur to image and save to disk
blurred_img = v2.ToPILImage()(gaussian_blur.squeeze(0))
blurred_img.save("./images/blur_output/fBlur.jpg")
