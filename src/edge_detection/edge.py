import torch
from torchvision.transforms import v2
import torch.nn as nn
from PIL import Image
import os

# Import image and convert to tensor
toTensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
img = toTensor(Image.open("./images/image1.jpg").convert("L")).unsqueeze(0)

# Define parameters
in_channels = 1
out_channels = 2
kernel_size = 3
stride = 1
padding = 0
bias = False

# Create Sobel kernel for edge detection
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda()
sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).cuda()

# Combine sobel tensors
kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)

# Mover tensors to GPU
img = img.cuda()
kernel = kernel.cuda()

# Create edge detection layer
edge_detection = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    bias=bias,
).cuda()
edge_detection.weight = nn.Parameter(kernel, requires_grad=False)

# Apply edge detection
edge_detection_img = edge_detection(img)

# Compute magnitude of edge detection
edge_detection_img = torch.norm(edge_detection_img, dim=1, keepdim=True)

# Clamp values between 0 and 1
edge_detection_img = torch.clamp(edge_detection_img, 0, 1)

# Create directory if it doesn't exist
if not os.path.exists("./images/edge_output"):
    os.makedirs("./images/edge_output")

# Save image
edge_detection_img = v2.ToPILImage()(edge_detection_img.squeeze(0)).convert("L")
edge_detection_img.save("./images/edge_output/edge_detection.jpg")
