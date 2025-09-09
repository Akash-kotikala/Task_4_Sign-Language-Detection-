import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd  # Unused, but kept if needed
import mediapipe as mp
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- Parameters ----------------
IMG_SIZE = 128
MODEL_PATH = "model.h5"

# ---------------- Load Model & Labels ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
labels = ["bye", "congratulations", "hello", "thankyou"]

# ---------------- Streamlit UI ----------------
st.title("🤟 Sign Language Detection")
st.write("Upload an image or use real-time webcam to detect hand gestures.")

option = st.radio("Choose Input Mode:", ["Upload Image", "Webcam"])

# ---------------- Image Upload Mode ----------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred_probs = model.predict(img_array, verbose=0)
        pred_class = np.argmax(pred_probs, axis=1)[0]
        pred_label = labels[pred_class]
        confidence = pred_probs[0][pred_class] * 100

        st.success(f"✅ Predicted Label: **{pred_label}** ({confidence:.2f}% confidence)")

# ---------------- Webcam Mode ----------------
elif option == "Webcam":
    st.write("Real-time hand gesture detection from webcam.")
    st.write("Press 'q' to quit the webcam window.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠ Unable to access webcam. Ensure it's connected and permissions are granted.")
        st.stop()

    frame_placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠ Unable to read from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box
                h, w, c = frame.shape
                x_min = w; y_min = h; x_max = y_max = 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Crop hand and preprocess
                margin = 20
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

                hand_crop = frame[y_min:y_max, x_min:x_max]
                if hand_crop.size != 0:
                    hand_img = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # Ensure RGB if model expects it
                    hand_img = hand_img / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)

                    pred_probs = model.predict(hand_img, verbose=0)
                    pred_class = np.argmax(pred_probs, axis=1)[0]
                    pred_label = labels[pred_class]
                    confidence = pred_probs[0][pred_class] * 100

                    cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)

        # Display in Streamlit (instead of cv2.imshow for web compatibility)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # For quitting, use a button in Streamlit or Ctrl+C in terminal
        if st.button("Stop Webcam"):
            break

    cap.release()
    cv2.destroyAllWindows()