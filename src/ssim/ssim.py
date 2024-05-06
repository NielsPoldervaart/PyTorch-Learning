import torch
from torchvision.transforms import v2
from PIL import Image


# TODO: Implement SSIM class
class SSIM:
    def __init__(self, window_size=11, sigma=1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.gaussian_window = GaussianWindow(window_size, sigma)

    def __call__(self, img1, img2):
        pass


# TODO: Implement Gaussian window for SSIM calculation
class GaussianWindow:
    def __init__(self, size, sigma):
        self.size = size
        self.sigma = sigma
        self.window = self._create_window()

    def _create_window(self):
        pass

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
