import streamlit as st
from ultralytics import YOLO
import cv2
import math
import numpy as np
import time

# Streamlit settings
st.title("VISION")
st.text("Real-time detection: Detecting if a person is real or fake using YOLOv8")

# Load YOLO model
model_path = "best.pt"  # Adjust to the correct path if needed
model = YOLO(model_path, task="detect")
classNames = ["fake", "real"]

# Confidence slider
confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.8)

# Lighting threshold slider
lighting_threshold = st.slider("Lighting threshold", min_value=0, max_value=100, value=50)

# Placeholder for video frame
stframe = st.empty()  # Placeholder for video frames
alert_placeholder = st.empty()  # Placeholder for alert message

# Function to check lighting
def check_lighting(frame, threshold=30):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)  # Calculate average brightness
    return brightness > threshold

# Function to process frame in real-time
def process_frame(img, confidence_threshold, lighting_threshold):
    # Check lighting conditions
    if check_lighting(img, lighting_threshold):
        flag = True
    else:
        flag = False
        st.warning("Insufficient Lighting - Please Adjust")
        return img, ""

    if flag:
        # Run YOLO model on the frame
        results = model(img, stream=True)
        alert_state = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Draw bounding boxes and text using OpenCV
                if conf > confidence_threshold:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    # Draw rectangle (bounding box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f'{classNames[cls].upper()} {int(conf * 100)}%'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_ymin = max(y1, label_size[1] + 10)
                    cv2.rectangle(img, (x1, label_ymin - label_size[1] - 10), 
                                  (x1 + label_size[0], label_ymin + 5), color, cv2.FILLED)
                    cv2.putText(img, label, (x1, label_ymin - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    res = classNames[cls].upper()

                    # Update alert state based on detection result
                    if res == 'FAKE':
                        alert_state = True
                    else:
                        alert_state = False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    if alert_state:
        return img, "Alert: Fake detected!"
    else:
        return img, ""

# Stream video frames in real-time
camera_input = st.camera_input("Turn on the camera for real-time detection")

if camera_input:
    while True:
        # Read the image as a numpy array for OpenCV
        bytes_data = camera_input.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Process the frame through YOLO and update the display
        img, alert_msg = process_frame(img, confidence_threshold, lighting_threshold)
        
        # Update the video frame
        stframe.image(img, channels="RGB", use_column_width=True)
        
        # Update the alert message
        if alert_msg:
            alert_placeholder.warning(alert_msg)
        else:
            alert_placeholder.empty()

        # Add a small delay to avoid overloading the browser
        time.sleep(0.03)
