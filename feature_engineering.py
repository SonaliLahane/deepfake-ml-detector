from skimage.feature import local_binary_pattern, hog
from skimage.io import imread
import numpy as np
import os

def extract_features(image_path):
    image = imread(image_path, as_gray=True)
    lbp = local_binary_pattern(image, P=8, R=1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return np.hstack([lbp_hist, hog_features])

def extract_features_from_folder(folder):
    features, labels = [], []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        for img in os.listdir(label_path):
            path = os.path.join(label_path, img)
            feat = extract_features(path)
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels)
