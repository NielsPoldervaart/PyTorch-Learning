import torch
from torchvision.transforms import v2
import torch.nn as nn
from PIL import Image
import os


class GaussianBlur:
    def __init__(self, sigma, channels, stride=1, bias=False):
        self.sigma = sigma
        self.kernel_size = int(6 * sigma + 1)
        self.kernel_size = (
            self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size
        )
        self.channels = channels
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.groups = channels
        self.bias = bias
        self.gaussian_blur = self.create_gaussian_blur_layer()

    def create_gaussian_blur_layer(self):
        ax = torch.arange(-self.kernel_size // 2 + 1.0, self.kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.channels, 1, 1, 1
        )

        gaussian_blur = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=self.bias,
        )
        gaussian_blur.weight.data = kernel
        gaussian_blur.weight.requires_grad = False

        return gaussian_blur

    def apply_blur(self, img):
        return self.gaussian_blur(img)

    @staticmethod
    def to_tensor(img):
        to_tensor_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        return to_tensor_transform(img)


if __name__ == "__main__":
    # Define parameters
    img = Image.open("./images/image1.jpg").convert("RGB")
    img_tensor = GaussianBlur.to_tensor(img).unsqueeze(0)
    channels = img_tensor.shape[1]
    sigma = 2.0
    stride = 1
    bias = False

    # Create GaussianBlur object and apply blur to image
    blur = GaussianBlur(sigma, channels, stride, bias)
    blurred_img = blur.apply_blur(img_tensor)

    # Create images folder if not exists
    if not os.path.exists("./images/blur_output"):
        os.makedirs("./images/blur_output")

    # Save image to disk
    blurred_img = v2.ToPILImage()(blurred_img.squeeze(0))
    blurred_img.save("./images/blur_output/nnBlur.jpg")
