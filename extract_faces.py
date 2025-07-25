import cv2
import os

def extract_faces_from_video(video_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/face_{count}.jpg", face)
            count += 1
    cap.release()
    return count
