import os
import glob
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Function to read and process an image
def read_image(image_path):
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Process the image as needed (e.g., resizing, normalization)
        # Example: image = cv2.resize(image, (desired_width, desired_height))
        
        return image
    except Exception as e:
        print(f"Error reading {image_path}: {str(e)}")
        return None

# Specify the main folder path
main_folder = "/home/adithya/tiny-imagenet-200/"

# List to store image paths and loaded images
image_paths = []
images = []

# Iterate through all subdirectories in the main folder
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith(".JPEG"):
            # Store the path to JPEG files in subdirectories
            jpeg_path = os.path.join(root, file)
            image_paths.append(jpeg_path)

# Use ThreadPoolExecutor for parallel image loading
num_threads = 4  # Adjust the number of threads as needed
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    images = list(executor.map(read_image, image_paths))

# Convert the list of images to a NumPy array
images_array = np.array(images)

print(len(images_array))
# Now, you have the images loaded and stored as a NumPy array (images_array).

