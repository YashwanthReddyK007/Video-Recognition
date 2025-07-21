import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import os

# -----------------------
# Title & Description
# -----------------------
st.set_page_config(page_title="Video/Image Recognition", layout="wide")
st.title("🎥 Video & 🖼️ Image Recognition App")
st.markdown("Upload an image or video, run detection with YOLOv8, and view the results!")

# -----------------------
# Load YOLO model
# -----------------------
@st.cache_resource
def load_model():
    # Latest ultralytics with pytorch 2.7.1
    model = YOLO("yolov8n.pt")  # or yolov8s.pt depending on accuracy/speed tradeoff
    return model

model = load_model()
st.success(f"✅ YOLOv8 model loaded (PyTorch {torch.__version__})")

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save to a temporary path
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"📁 Saved file to {file_path}")

    # -----------------------
    # Process Image
    # -----------------------
    if uploaded_file.type.startswith("image"):
        # Display original
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run detection
        st.write("🔍 Running detection… please wait")
        results = model(image)
        # Save the annotated result
        result_img_path = os.path.join(temp_dir, "result.jpg")
        annotated_img = results[0].plot()  # numpy array
        Image.fromarray(annotated_img).save(result_img_path)

        st.image(annotated_img, caption="Detection Result", use_container_width=True)

        # Summary of detections
        names = model.names
        detected_counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = names.get(cls_id, str(cls_id))
            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1

        st.markdown("### 📋 Detection Summary")
        if detected_counts:
            for k, v in detected_counts.items():
                st.write(f"✅ {k}: {v}")
        else:
            st.write("⚠️ No objects detected.")

    # -----------------------
    # Process Video
    # -----------------------
    elif uploaded_file.type.startswith("video"):
        st.video(file_path)
        st.write("🔍 Running detection on video… this may take a while.")

        # Run YOLO prediction on video (stream=True allows frame-by-frame processing)
        output_path = os.path.join(temp_dir, "result_video.mp4")
        results = model.predict(file_path, save=True, project=temp_dir, name="video_results")
        # YOLO saves processed video in temp_dir/video_results
        predicted_video = os.path.join(temp_dir, "video_results", uploaded_file.name)

        if os.path.exists(predicted_video):
            st.video(predicted_video)
            st.success("✅ Detection completed on video!")
        else:
            st.error("❌ Processed video not found. Check logs for errors.")
