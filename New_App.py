import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Streamlit settings
st.title("VISION")
st.text("This app detects if a person is real or fake using YOLOv8")

# Load YOLO model
model_path = "best.onnx"
model = YOLO(model_path, task="detect")
classNames = ["fake", "real"]

# Confidence slider
confidence_threshold = 0.8

# Lighting threshold slider
lighting_threshold = 50

# Function to check lighting
def check_lighting(frame, threshold=30):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)  # Calculate average brightness
    return brightness > threshold

# Video Transformer using Streamlit WebRTC
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.alert_state = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Check lighting condition
        flag = check_lighting(img, lighting_threshold)

        if flag:
            # Run YOLO model on the frame
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    # Draw bounding boxes and text
                    if conf > confidence_threshold:
                        color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                        cvzone.putTextRect(
                            img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                            (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                            colorB=color
                        )
                        res = classNames[cls].upper()

                        # Update alert state based on result
                        if res == 'FAKE':
                            self.alert_state = True
                        elif res == 'REAL':
                            self.alert_state = False

            # Calculate FPS
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

        # Display alert if needed
        if self.alert_state:
            st.warning("Alert: Fake detected!")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Use streamlit-webrtc for webcam access
webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

# Display lighting condition
st.sidebar.title("Lighting and Confidence Settings")
st.sidebar.markdown(f"**Lighting Threshold:** {lighting_threshold}")
st.sidebar.markdown(f"**Confidence Threshold:** {confidence_threshold * 100}%")
