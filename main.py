from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from a file path and convert it to a numpy array."""
    image = Image.open(image_path)
    return np.array(image)

image = load_image('images/lenna.png')


print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image size:", image.size)

print(image[0])

image_normalized = (image - np.mean(image)) / np.std(image)

print(image_normalized[0])