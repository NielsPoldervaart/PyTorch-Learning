from PIL import Image
import os

# Create images folder if not exists
if not os.path.exists("./images/pixel_changer_output"):
    os.makedirs("./images/pixel_changer_output")

# Load image
img = Image.open("./images/image1.jpg").convert("RGB")

# Change pixel
pixel = img.load()
pixel[-83, -2] = (255, 255, 0)

# Save image
# Save as png to avoid loss of quality
img.save("./images/pixel_changer_output/image_new_negative.png")
