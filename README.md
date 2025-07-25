# DeepFake Detection using AI/ML (Face-Swap Focused)

This project aims to detect face-swap-based deepfake videos using traditional AI/ML techniques instead of relying solely on deep learning.

## Features
- Face detection using OpenCV
- Feature extraction (LBP, HOG)
- Classification using SVM / Random Forest
- Frame-wise prediction and final verdict
- Streamlit-based user interface

## Project Structure
- `app.py`: Streamlit frontend to upload video and see prediction
- `extract_faces.py`: Extracts faces from video frames
- `feature_engineering.py`: Extracts features like LBP and HOG
- `train_model.py`: Trains ML classifier on extracted features
- `predict.py`: Uses the model to predict if video is deepfake
- `utils.py`: Utility functions
- `requirements.txt`: Required libraries

## Run the app
```bash
streamlit run app.py
```

## Model
Use `train_model.py` with your dataset to generate a model, or use our sample model under `models/`.
