# preprocess.py
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def z_score_normalize(image):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image) + 1e-8
    return (image - mean) / std

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = apply_clahe(image)
    image = z_score_normalize(image)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image
