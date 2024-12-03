import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set page config
st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("Construction Site PPE Detection System")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt', task='detect')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image):
    try:
        results = model.predict(source=image, device='cpu', conf=0.25)
        return results[0].plot()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(source=frame, device='cpu', conf=0.25)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame_rgb)
            
        cap.release()
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# Initialize model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check your installation.")
    st.stop()

# Sidebar
st.sidebar.title("Upload Media")
media_type = st.sidebar.radio("Select Media Type", ["Image", "Video"])

if media_type == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.subheader("Detection Results")
        processed_image = process_image(image)
        if processed_image is not None:
            st.image(processed_image, channels="BGR", use_column_width=True)

elif media_type == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.subheader("Live Detection")
        process_video(tfile.name)
        
        os.unlink(tfile.name)

# Display instructions
st.sidebar.markdown("""
### Instructions:
1. Select media type (Image/Video)
2. Upload your file
3. Wait for detection results
4. For video, detection will be shown in real-time
""")

# Display PPE requirements
st.sidebar.markdown("""
### Required PPE:
- Safety Helmet
- Safety Gloves
- Safety Goggles
- Safety Shoes
- Reflective Jacket
""")
