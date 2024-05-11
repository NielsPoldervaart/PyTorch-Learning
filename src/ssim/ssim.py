import torch
from torchvision.transforms import v2
from torch.nn import functional as F
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt

# TODO: Possibly implement MS-SSIM as well


# TODO: Implement SSIM class
class SSIM(torch.nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        window_sigma: float = 1.5,
        channels: int = 3,
        size_average: bool = True,
        full: bool = False,
        window_type: str = "2D",
        device: str = "cpu",
    ) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.window_sigma: float = window_sigma
        self.channels: int = channels
        self.size_average: bool = size_average
        self.full: bool = full
        self.device: str = device

        if window_type == "1D":
            self.window: Tensor = GaussianWindow1D(device, window_size, window_sigma)()
        elif window_type == "2D":
            self.window: Tensor = GaussianWindow2D(
                device, window_size, window_sigma, channels
            )()
        else:
            raise ValueError("Window type must be either '1D' or '2D'")

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        # Calculate the mean (average) of the images
        mu1 = F.conv2d(
            img1, self.window, padding=self.window_size // 2, groups=self.channels
        )
        mu2 = F.conv2d(
            img2, self.window, padding=self.window_size // 2, groups=self.channels
        )

        # Squares of the images
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        # Product of the images
        mu1_mu2 = mu1 * mu2

        # Calculate the variance of the images
        sigma1_sq = (
            F.conv2d(
                img1 * img1,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu2_sq
        )

        # Calculate the covariance of the images
        sigma12 = (
            F.conv2d(
                img1 * img2,
                self.window,
                padding=self.window_size // 2,
                groups=self.channels,
            )
            - mu1_mu2
        )

        # Constants for stability
        # TODO: Use dynamic range of the images (1 or 255) based on the input images
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Calculate the SSIM map
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        # return the SSIM value
        if self.size_average:
            return ssim_map.mean()
        elif self.full:
            return ssim_map
        else:
            # Return the mean SSIM value for each channel (channels, height, width)
            return ssim_map.mean([1, 2, 3])


# Create a 1D Gaussian window for SSIM calculation
class GaussianWindow1D:
    def __init__(self, device: str, size: int, sigma: float) -> None:
        self.size: int = size
        self.sigma: float = sigma
        self.window: Tensor = self._create_window().to(device)

    def _create_window(self) -> Tensor:
        axis: Tensor = torch.arange(-self.size // 2 + 1.0, self.size // 2 + 1.0)
        kernel: Tensor = torch.exp(-(axis**2) / (2.0 * self.sigma**2))
        kernel /= torch.sum(kernel)
        kernel = kernel.view(1, 1, self.size, 1).repeat(1, 1, 1, 1)

        return kernel

    def __call__(self) -> Tensor:
        return self.window


# Create a 2D Gaussian window for SSIM calculation
class GaussianWindow2D:
    def __init__(self, device: str, size: int, sigma: float, channels: int = 3) -> None:
        self.size: int = size
        self.sigma: float = sigma
        self.channels: int = channels
        self.window: Tensor = self._create_window().to(device)

    def _create_window(self) -> Tensor:
        axis: Tensor = torch.arange(-self.size // 2 + 1.0, self.size // 2 + 1.0)
        xx: Tensor
        yy: Tensor
        xx, yy = torch.meshgrid(axis, axis, indexing="xy")
        kernel: Tensor = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel /= torch.sum(kernel)
        kernel = kernel.view(1, 1, self.size, self.size).repeat(self.channels, 1, 1, 1)

        return kernel

    def __call__(self) -> Tensor:
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


def visualize_kernels(device: str):
    # Create a 1D Gaussian window
    window1D = GaussianWindow1D(device="cpu", size=11, sigma=1.5)
    kernel1D = window1D()

    # Create a 2D Gaussian window
    window2D = GaussianWindow2D(device="cpu", size=11, sigma=1.5)
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
    ssim_module = SSIM(window_size=11, window_sigma=1.5, device=device).to(device)
    result = ssim_module(img1, img2)

    # Print final result as percentage
    print(f"Result: {result.item() * 100:.2f}%")

    # Visualize 1D and 2D Gaussian kernels
    visualize_kernels(device)
