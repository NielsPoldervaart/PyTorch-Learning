import torch
from torchvision.transforms import v2
from torch import Tensor
from PIL import Image

# TODO: Possibly implement MS-SSIM as well


# TODO: Implement SSIM class
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        self.window_size: float = window_size
        self.window_sigma: float = window_sigma
        self.k1: float = k1
        self.k2: float = k2

    def __call__(self, img1, img2):
        pass

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        return ssim()


# TODO: Implement Gaussian window for SSIM calculation
class GaussianWindow:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.window = self._create_window()

    def _create_window(self):
        pass

    def __call__(self):
        pass


def ssim(img1, img2, window_size=11, window_sigma=1.5, k1=0.01, k2=0.03):
    # Calculate SSIM
    pass


def preprocess_img(img_path):
    preprocess = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),  # scale=False for [0, 255] range
        ]
    )

    return preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)


if __name__ == "__main__":
    # Check if CUDA is available and set PyTorch to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess images
    img1 = preprocess_img("./images/image1.jpg").to(device)
    img2 = preprocess_img("./images/image2.jpg").to(device)

    # Calculate SSIM
    result = SSIM()
    # TODO: Calculate % similarity between the two images

    # Print final result
    print(f"Result: {result}")
