import streamlit as st
from predict import predict_video

st.title("Face-Swap Deepfake Video Detector (ML-based)")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)
    result = predict_video(uploaded_file)
    st.markdown(f"**Prediction:** {'🟥 Deepfake' if result else '🟩 Authentic'}")
