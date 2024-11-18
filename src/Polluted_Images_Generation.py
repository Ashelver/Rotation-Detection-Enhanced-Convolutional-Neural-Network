import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Custom Random Rotation and Noisy With EdgePadding
class CRRNWEP:
    def __init__(self, range1=(-30, -15), range2=(15, 30), size=(28, 28), noise_std=0.):
        self.range1 = range1
        self.range2 = range2
        self.size = size  # Set the target size
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

    def calculate_edge_color(self, img):
        # Get the edge pixels of the image
        width, height = img.size
        edge_pixels = []

        # Collect pixels from the four edges of the image
        # Top edge
        edge_pixels.extend(np.array(img.crop((0, 0, width, 1))).flatten())
        # Bottom edge
        edge_pixels.extend(np.array(img.crop((0, height-1, width, height))).flatten())
        # Left edge
        edge_pixels.extend(np.array(img.crop((0, 0, 1, height))).flatten())
        # Right edge
        edge_pixels.extend(np.array(img.crop((width-1, 0, width, height))).flatten())

        # Calculate the mean color of the edge pixels
        edge_pixels = np.array(edge_pixels)
        mean_color = edge_pixels.mean(axis=0)  # Get the mean value for each channel

        # If mean_color is a scalar, return it as a single-value tuple
        if isinstance(mean_color, np.ndarray):
            # Ensure the result is a tuple of integers (for each channel if applicable)
            return tuple(mean_color.astype(int).tolist())
        else:
            # If the result is a scalar, return it in a tuple form
            return (int(mean_color),)

    def add_gaussian_noise(self, img):
        """Adds Gaussian noise to the image."""
        # Convert the image to a numpy array
        img_np = np.array(img).astype(np.float32)  # Ensure float type for noise addition
        
        # Generate Gaussian noise
        noise = np.random.normal(0, self.noise_std*255, img_np.shape)  # Mean = 0, std = noise_std
        
        # Add noise to the image
        noisy_img_np = img_np + noise
        
        # Clip the noisy image to ensure the pixel values are within valid range [0, 255]
        noisy_img_np = np.clip(noisy_img_np, 0, 255)
        
        # Convert back to PIL image
        noisy_img = Image.fromarray(np.uint8(noisy_img_np))
        return noisy_img


    def __call__(self, img):
        # Randomly choose a rotation angle within the specified ranges
        angle = random.choice([random.uniform(*self.range1), random.uniform(*self.range2)])

        # Rotate the image
        img = F.rotate(img, angle)

        # Get the size of the rotated image
        width, height = img.size

        # Calculate the edge color for padding
        edge_color = self.calculate_edge_color(img)

        # Calculate the padding needed to maintain the original image size
        left = (width - self.size[0]) // 2
        top = (height - self.size[1]) // 2
        right = width - self.size[0] - left
        bottom = height - self.size[1] - top

        # Apply padding to the image using the calculated edge color
        img = F.pad(img, (left, top, right, bottom), fill=edge_color)

        # Add Gaussian noise to the image
        img = self.add_gaussian_noise(img)

        return img


def Show_Examples():

    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Original data
    original_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # Define the transform with normalization and rotation
    rotating_transform = transforms.Compose([
        CRRNWEP(range1=(-30, -10), range2=(10, 30), size=(28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    # Anti-normalize for visualization
    inv_normalize = transforms.Normalize(mean=(-0.5 / 0.5,), std=(1 / 0.5,))

    # Load the FashionMNIST dataset
    original_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=None)

    test_size = len(original_dataset)

    print(f"Testing dataset size: {test_size}")


    # Generate 6 rotated images for visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(3):
        # Get the image and label
        original_image, label = original_dataset[i]
        
        # Apply rotation transform (ensure it returns a PIL image after processing)
        rotated_image = transforms.ToPILImage()(inv_normalize(rotating_transform(original_image)))  # Undo normalization and convert tensor to PIL image
        
        label = label  # Get the label
        
        # Original image
        axes[i].imshow(original_image,cmap='gray')
        axes[i].set_title(f"Original Image (Label: {labels[label]})")
        axes[i].axis('off')

        # Rotated image
        axes[i + 3].imshow(rotated_image,cmap='gray')
        axes[i + 3].set_title(f"Rotated Image (Label: {labels[label]})")
        axes[i + 3].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Show_Examples()