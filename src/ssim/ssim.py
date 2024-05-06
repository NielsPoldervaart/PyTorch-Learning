import torch
from torchvision.transforms import v2
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt

# TODO: Possibly implement MS-SSIM as well


# TODO: Implement SSIM class
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size: float = window_size
        self.window_sigma: float = window_sigma

    # TODO: Implement forward method (SSIM calculation)
    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        pass


# Create a 1D Gaussian window for SSIM calculation
class GaussianWindow1D:
    def __init__(self, size, sigma):
        self.size: float = size
        self.sigma: float = sigma
        self.window = self._create_window()

    def _create_window(self):
        axis = torch.arange(-self.size // 2 + 1.0, self.size // 2 + 1.0)
        kernel = torch.exp(-(axis**2) / (2.0 * self.sigma**2))
        kernel /= torch.sum(kernel)
        kernel = kernel.view(1, 1, self.size, 1).repeat(1, 1, 1, 1)

        return kernel

    def __call__(self):
        return self.window


# Create a 2D Gaussian window for SSIM calculation
class GaussianWindow2D:
    def __init__(self, size, sigma, channels=3):
        self.size: float = size
        self.sigma: float = sigma
        self.channels: float = channels
        self.window = self._create_window()

    def _create_window(self):
        axis = torch.arange(-self.size // 2 + 1.0, self.size // 2 + 1.0)
        xx, yy = torch.meshgrid(axis, axis)
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel /= torch.sum(kernel)
        kernel = kernel.view(1, 1, self.size, self.size).repeat(self.channels, 1, 1, 1)

        return kernel

    def __call__(self):
        return self.window


def preprocess_img(img_path):
    preprocess = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),  # scale=False for [0, 255] range
        ]
    )

    return preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)


def visualize_kernels():
    # Create a 1D Gaussian window
    window1D = GaussianWindow1D(size=11, sigma=1.5)
    kernel1D = window1D()

    # Create a 2D Gaussian window
    window2D = GaussianWindow2D(size=11, sigma=1.5)
    kernel2D = window2D()

    # Plot the 1D Gaussian kernel
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(kernel1D.view(-1).numpy())
    plt.title("1D Gaussian Kernel")

    # Plot the 2D Gaussian kernel
    plt.subplot(1, 2, 2)
    plt.imshow(
        kernel2D[0, 0].numpy(),
        cmap="hot",
        interpolation="nearest",
    )
    plt.title("2D Gaussian Kernel")

    plt.show()


if __name__ == "__main__":
    # Check if CUDA is available and set PyTorch to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess images
    img1 = preprocess_img("./images/image1.jpg").to(device)
    img2 = preprocess_img("./images/image2.jpg").to(device)

    # Calculate SSIM
    ssim_module = SSIM(window_size=11, window_sigma=1.5).to(device)
    result = 1 - SSIM(img1, img2)
    # TODO: Calculate % similarity between the two images

    # Print final result
    print(f"Result: {result}")

    # Visualize 1D and 2D Gaussian kernels
    visualize_kernels()
