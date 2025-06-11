import streamlit as st
import cv2
import tempfile
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

# Load model
model = YOLO("my_model.pt")
labels = model.names

nutrition_info = {
    'SnowBear': [23, 6],
    'Maxx': [15, 3],
    'Frutos': [10, 2],
    'Mentos': [10, 2],
    'X.O': [15, 2]
}

st.set_page_config(page_title="Candy Calorie Counter", layout="centered")
st.title("🍬 Candy Calorie Counter")

# Session state init
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "video_source" not in st.session_state:
    st.session_state.video_source = None
if "video_total_frames" not in st.session_state:
    st.session_state.video_total_frames = 0

# Source selection
source = st.radio("Choose video source", ("Webcam", "Upload Video"))

# Handle source setup only when not detecting
if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file and not st.session_state.detecting:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.cap = cv2.VideoCapture(tfile.name)
        st.session_state.video_source = "upload"
        st.session_state.video_total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    if not st.session_state.detecting:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.video_source = "webcam"
        st.session_state.video_total_frames = 0

# Smart Start/Stop toggle
if st.button("⏹️ Stop Detection" if st.session_state.detecting else "▶️ Start Detection"):
    st.session_state.detecting = not st.session_state.detecting
    if not st.session_state.detecting and st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.rerun()  # force UI update

# Frame processor
def process_frame(frame):
    results = model(frame, verbose=False)[0]
    detections = results.boxes
    candy_counts = defaultdict(int)

    for det in detections:
        conf = det.conf.item()
        if conf > 0.5:
            cls_id = int(det.cls.item())
            name = labels[cls_id]
            candy_counts[name] += 1
            x1, y1, x2, y2 = map(int, det.xyxy.squeeze().tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {int(conf*100)}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    total_candies = sum(candy_counts.values())
    total_calories = sum(nutrition_info[c][0] * count for c, count in candy_counts.items())
    total_sugar = sum(nutrition_info[c][1] * count for c, count in candy_counts.items())

    overlay = frame.copy()
    box_h = 130 + 20 * len(candy_counts)
    cv2.rectangle(overlay, (10, 10), (280, box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    y = 30
    cv2.putText(frame, f"Total Candies: {total_candies}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    cv2.putText(frame, f"Calories: {total_calories}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 255, 153), 2)
    y += 25
    cv2.putText(frame, f"Sugar: {total_sugar}g", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (153, 204, 255), 2)
    y += 30
    for c in candy_counts:
        cv2.putText(frame, f"{c}: {candy_counts[c]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 204, 102), 2)
        y += 25

    return frame

# Placeholders
frame_placeholder = st.empty()
fps_placeholder = st.empty()
progress_bar = st.empty()

# Detection loop
if st.session_state.detecting and st.session_state.cap:
    cap = st.session_state.cap
    frame_count = 0
    total_frames = st.session_state.video_total_frames

    while cap.isOpened() and st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (720, 480))

        start = time.time()
        result = process_frame(frame)
        end = time.time()
        fps = 1 / (end - start + 1e-5)

        frame_placeholder.image(result, channels="BGR", use_container_width=True)
        fps_placeholder.markdown(f"**FPS:** {fps:.2f}")
        frame_count += 1

        if st.session_state.video_source == "upload" and total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    st.session_state.cap = None
    st.session_state.detecting = False
    frame_placeholder.empty()
    fps_placeholder.empty()
    progress_bar.empty()
