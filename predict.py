from extract_faces import extract_faces_from_video
from feature_engineering import extract_features
import joblib
import os

def predict_video(video_file):
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    face_count = extract_faces_from_video("temp_video.mp4", output_dir="temp_faces")

    clf = joblib.load("models/deepfake_detector.pkl")

    votes = []
    for img in os.listdir("temp_faces"):
        feat = extract_features(os.path.join("temp_faces", img))
        pred = clf.predict([feat])
        votes.append(pred[0])

    return votes.count("deepfake") > votes.count("real")
