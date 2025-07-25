# generate_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import os

# Step 1: Create dummy classification data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Step 2: Train a simple RandomForest model
clf = RandomForestClassifier()
clf.fit(X, y)

# Step 3: Save the trained model to 'models' directory
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/deepfake_detector.pkl")

print("✅ Model saved to models/deepfake_detector.pkl")
