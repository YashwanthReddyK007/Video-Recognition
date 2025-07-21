import streamlit as st
import tempfile
import os
import cv2
import json
import faiss
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
from collections import Counter

# =======================
# SETTINGS
# =======================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# =======================
# Load models (cached)
# =======================
@st.cache_resource
def load_yolo():
    # load architecture and pretrained weights
    m = YOLO("yolov8n.yaml")  # architecture file
    m.load("yolov8n.pt")      # weights
    return m

yolo_model = load_yolo()

@st.cache_resource
def load_clip():
    m, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return m.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-B-32")

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# Helper functions
# =======================
def extract_frames(video_path, every_n=30):
    """Extract frames from a video every N frames and save in FRAMES_FOLDER."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            cv2.imwrite(fpath, frame)
            saved.append(fname)
        count += 1
    cap.release()
    return saved

def detect_objects(img_path):
    """Run YOLO detection on an image path."""
    results = yolo_model(img_path, conf=0.3, verbose=False)
    objs = []
    for box in results[0].boxes:
        objs.append({
            "label": results[0].names[int(box.cls[0])],
            "conf": float(box.conf[0])
        })
    return objs

def encode_image(img_path):
    """Encode an image to a CLIP embedding."""
    img = Image.open(img_path).convert("RGB")
    tens = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Video Intelligence", layout="wide")
st.title("🎥🔎 Video Intelligence with YOLO + CLIP")

# ============ Video Upload & Processing ============
uploaded_videos = st.file_uploader(
    "Upload video files",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True
)

if uploaded_videos:
    st.info("⏳ Processing uploaded videos...")
    all_frames = []
    all_objects = {}
    all_embeddings = []

    for video in uploaded_videos:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.name)[1]) as tmp:
            tmp.write(video.read())
            tmp_path = tmp.name

        st.write(f"📥 Extracting frames from **{video.name}**...")
        frames = extract_frames(tmp_path, every_n=30)
        all_frames.extend(frames)

        for frame in frames:
            fpath = os.path.join(FRAMES_FOLDER, frame)
            objs = detect_objects(fpath)
            all_objects[frame] = objs
            emb = encode_image(fpath)
            all_embeddings.append(emb)

    # Save metadata
    with open("metadata.txt", "w") as f:
        for m in all_frames:
            f.write(f"{m}\n")
    with open("objects.json", "w") as f:
        json.dump(all_objects, f, indent=2)

    # Build FAISS index
    if all_embeddings:
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "video_index.faiss")
        st.success("✅ Processing complete! You can now search or summarize the video.")
    else:
        st.error("⚠️ No frames processed!")

st.markdown("---")
st.header("🔎 Search in processed videos")

# ============ Search & Summarize ============
if os.path.exists("video_index.faiss") and os.path.exists("metadata.txt") and os.path.exists("objects.json"):
    query = st.text_input("Enter a text description to search:")
    if query:
        index = faiss.read_index("video_index.faiss")
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)

        st.subheader(f"Top matches for **'{query}'**:")
        for idx, score in zip(I[0], D[0]):
            fname = metadata[idx]
            img_path = os.path.join(FRAMES_FOLDER, fname)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{fname} (score {score:.3f})", use_column_width=True)
            else:
                st.write(f"❌ Image file not found: {img_path}")
            st.write("Objects detected in this frame:")
            for obj in objects_map.get(fname, []):
                st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")

    st.markdown("---")
    st.header("📋 Summarize Video Content")
    if st.button("Summarize All Objects"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected yet.")
        else:
            counts = Counter(all_labels)
            st.write("✅ **Summary of objects detected across all frames:**")
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} frames")

else:
    st.warning("⚠️ Please upload and process videos first.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, YOLOv8, and CLIP")
