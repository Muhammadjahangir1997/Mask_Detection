import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"  # Ye rakh sakte ho, lekin fix ke baad shayad na bhi lage
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2  # Ye add karna zaroori hai

st.set_page_config(page_title="Mask Detection", layout="centered")
st.title("😷 Mask Detection App")
st.write("Upload an image to detect mask")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            # FIX YAHAN HAI
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = model(img_bgr)
            
            result_img_bgr = results[0].plot()  # Ye BGR mein hi hota hai
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)  # Streamlit ke liye RGB
            
            st.image(result_img_rgb, caption="Detection Result")
            
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"Detections Found: {len(boxes)}")
            else:
                st.warning("No object detected")
