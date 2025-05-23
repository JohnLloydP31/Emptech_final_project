import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile
from PIL import ImageOps

# Load the trained model
model = YOLO("best.pt")

st.title("Fatigue Detection with YOLO")
st.write("Group Members:.")
st.write("ANGELO DAN RENZ DELLOSON")
st.write("Jeremy John Orlina")
st.write("John LloydPadrigano")
st.write("Upload an image to detect 'awake' or 'fatigued' faces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Convert to NumPy array
    image_np = np.array(image)

    # Inference
    results = model(image_np)

    # Draw results on image
    annotated_frame = results[0].plot()

    # Display result
    st.image(annotated_frame, caption='Detected', use_container_width=True)
