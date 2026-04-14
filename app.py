import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("cancer_model.h5")

IMG_SIZE = 224

# 🌈 LIGHT COLORFUL BACKGROUND
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #e1bee7);
    }

    .main-container {
        background-color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }

    h1 {
        text-align: center;
        color: #4a148c;
        font-family: 'Segoe UI';
    }

    .upload-box {
        border: 2px dashed #7b1fa2;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: #f3e5f5;
    }

    .stButton>button {
        background: linear-gradient(to right, #7b1fa2, #4dd0e1);
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🧠 Cancer Detection using Deep Learning</h1>", unsafe_allow_html=True)

st.write("### Upload an MRI image to detect tumor presence")

st.sidebar.title("🧠 About")
st.sidebar.info("AI-based brain tumor detection system using CNN.")
st.sidebar.info("Upload MRI image and this will detect the presence or absence of tumor in the image.")
# Upload section
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img_resized)

    if prediction > 0.5:
        st.error("⚠️ Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")