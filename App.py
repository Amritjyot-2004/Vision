import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import time

# Streamlit settings
st.title("VISION")
st.text("This app detects if a person is real or fake using YOLOv8")

# Load YOLO model
model_path = "best.onnx"
model = YOLO(model_path)
classNames = ["fake", "real"]

# Confidence slider
confidence_threshold = .8

# Lighting threshold slider
lighting_threshold = 50

# Open webcam (set up only once)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Function to check lighting
def check_lighting(frame, threshold=30):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)  # Calculate average brightness
    return brightness > threshold

prev_frame_time = 0
new_frame_time = 0

# Initialize the alert state and create a placeholder
alert_state = False
alert_placeholder = st.empty()  # Placeholder for the alert message

# Stream video and run YOLO model
stframe = st.empty()  # Placeholder for video frames

while cap.isOpened():
    success, img = cap.read()
    
    if not success:
        st.warning("Failed to capture video. Make sure your webcam is connected.")
        break

    new_frame_time = time.time()
    if check_lighting(img, lighting_threshold):
        flag = True
    else:
        flag = False
        #st.error("Insufficient Lighting - Please Adjust")

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

                # Draw the bounding boxes and text
                if conf > confidence_threshold:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                       colorB=color)
                    res = classNames[cls].upper()
                    #st.text(res)
                    
                    # Update alert state based on res
                    if res == 'FAKE':
                        alert_state = True
                    elif res == 'REAL':
                        alert_state = False

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

    # Display the image on the Streamlit app
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    stframe.image(imgRGB, channels="RGB", use_column_width=True)

    # Update the alert message in the fixed position
    if alert_state:
        alert_placeholder.warning("Alert: Fake detected!")
    else:
        alert_placeholder.empty()  # Clear the placeholder if no alert
    

# Release the video capture
cap.release()
