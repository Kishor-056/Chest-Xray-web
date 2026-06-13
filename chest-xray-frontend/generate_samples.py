
import numpy as np
from PIL import Image
import cv2
import os

def create_valid_xray(filename):
    # Create grayscale image 500x500
    width, height = 500, 500
    # Create a base gray background
    img_array = np.full((height, width, 3), 128, dtype=np.uint8)
    
    # Add some "ribs" (white stripes) to simulate edges/content for edge density check
    for i in range(50, 450, 50):
        cv2.line(img_array, (50, i), (450, i), (200, 200, 200), 10)
    
    # Add a "heart" (darker area)
    cv2.circle(img_array, (300, 300), 50, (50, 50, 50), -1)
    
    # Ensure it's perfectly grayscale (R=G=B)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"Created valid image: {filename}")

def create_invalid_image(filename):
    # Create colorful image
    width, height = 500, 500
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red background
    img_array[:] = (255, 0, 0)
    
    # Green and Blue shapes
    cv2.rectangle(img_array, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.circle(img_array, (300, 300), 50, (0, 0, 255), -1)
    
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"Created invalid image: {filename}")

if __name__ == "__main__":
    os.makedirs("public/samples", exist_ok=True)
    create_valid_xray("public/samples/valid_xray.png")
    create_invalid_image("public/samples/invalid_image.png")
