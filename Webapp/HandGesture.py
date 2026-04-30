import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("MAINTRAINING.keras")

# Class names (EDIT THIS to match your dataset)
class_names = ["Peace Sign", "Closed Fist", "Open Palm", "Point Finger", "Thumbs Up"]

st.title("🖐️ Hand Gesture Recognition App")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera error")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Preprocess
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    label = f"{class_names[class_idx]} ({confidence*100:.2f}%)"

    # Show label on frame
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()