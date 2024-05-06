import torch
from torchvision.transforms import v2
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model and move to GPU
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
# weights = VGG16_Weights.DEFAULT
# model = vgg16(weights=weights)
# print(model)
model = model.to(device)
model = model.eval()

# Define the image transformations
preprocess = v2.Compose(
    [
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the images
image1 = Image.open("./images/image9.jpg").convert("RGB")
image2 = Image.open("./images/image10.jpg").convert("RGB")

# Preprocess the images and move them to the GPU
input1 = preprocess(image1).unsqueeze(0).to(device)
input2 = preprocess(image2).unsqueeze(0).to(device)

# Pass the images through the model
with torch.no_grad():
    features1 = model(input1)
    features2 = model(input2)

# Normalize the feature vectors
features1 = features1 / torch.norm(features1)
features2 = features2 / torch.norm(features2)

# Calculate the Euclidean distance manually
distance = torch.sqrt(torch.sum((features1 - features2) ** 2))

print(f"Distance: {distance.item()}")

# Convert to similarity score
max_distance = torch.sqrt(torch.tensor(2.0)).to(device)
similarity = 1 - (distance / max_distance)
similarity_map = similarity.squeeze().cpu().numpy()
similarity = similarity * 100

print(f"Similarity: {round(similarity.item(), 2)}%")

# Calculate Euclidean distance
distance = torch.dist(features1, features2)

print(f"Distance using dist: {distance.item()}")
