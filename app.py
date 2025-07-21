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

# =======================
# SETTINGS
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="Video Search (YOLO + CLIP)", layout="wide")
st.title("🎥🔎 Video Search with YOLOv8 + CLIP")

# Use a session-local frames directory in temp
FRAMES_DIR = os.path.join(tempfile.gettempdir(), "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# =======================
# LOAD MODELS
# =======================
@st.cache_resource
def load_yolo():
    # lightweight YOLO model for cloud
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# SAVE METADATA & INDEX
# =======================
@st.cache_data
def save_metadata(frames, objects, embeddings):
    # store metadata in temp directory
    meta_dir = tempfile.gettempdir()
    with open(os.path.join(meta_dir, "metadata.txt"), "w") as f:
        for fname in frames:
            f.write(f"{fname}\n")
    with open(os.path.join(meta_dir, "objects.json"), "w") as f:
        json.dump(objects, f, indent=2)

    dim = embeddings[0].shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.vstack(embeddings))
    faiss.write_index(index, os.path.join(meta_dir, "video_index.faiss"))

# =======================
# HELPERS
# =======================
def extract_frames(video_path, every_n=30):
    """Extract frames to FRAMES_DIR"""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = f"{os.path.basename(video_path)}_{count}.jpg"
            fpath = os.path.join(FRAMES_DIR, fname)
            cv2.imwrite(fpath, frame)
            saved.append(fname)
        count += 1
    cap.release()
    return saved

def detect_objects(img_path):
    results = yolo_model(img_path, conf=0.3, verbose=False)
    return [
        {"label": results[0].names[int(box.cls[0])], "conf": float(box.conf[0])}
        for box in results[0].boxes
    ]

def encode_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img_tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

# =======================
# VIDEO UPLOAD & PROCESS
# =======================
uploaded_videos = st.file_uploader(
    "Upload video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True
)

if uploaded_videos:
    with st.spinner("⏳ Processing uploaded videos..."):
        all_frames, all_objects, all_embeddings = [], {}, []

        for video in uploaded_videos:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.name)[1]) as tmp:
                tmp.write(video.read())
                tmp_path = tmp.name

            st.write(f"📥 Extracting frames from **{video.name}**...")
            frames = extract_frames(tmp_path, every_n=30)
            all_frames.extend(frames)

            for frame in frames:
                fpath = os.path.join(FRAMES_DIR, frame)
                objs = detect_objects(fpath)
                emb = encode_image(fpath)
                all_objects[frame] = objs
                all_embeddings.append(emb)

        if all_embeddings:
            save_metadata(all_frames, all_objects, all_embeddings)
            st.success("✅ Processing complete! You can now search for frames.")
        else:
            st.error("⚠️ No frames processed!")

# =======================
# SEARCH SECTION
# =======================
st.markdown("---")
st.header("🔎 Search in processed frames")

meta_dir = tempfile.gettempdir()
index_path = os.path.join(meta_dir, "video_index.faiss")
meta_txt_path = os.path.join(meta_dir, "metadata.txt")
objects_json_path = os.path.join(meta_dir, "objects.json")

if os.path.exists(index_path):
    query = st.text_input("Enter a text description to search:")
    if query:
        index = faiss.read_index(index_path)
        metadata = [line.strip() for line in open(meta_txt_path)]
        objects_map = json.load(open(objects_json_path))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)
        st.subheader(f"Top matches for **'{query}'**:")

        for idx, score in zip(I[0], D[0]):
            fname = metadata[idx]
            img_path = os.path.join(FRAMES_DIR, fname)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{fname} (score {score:.3f})", use_container_width=True)
                st.write("Detected objects:")
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
            else:
                st.write(f"❌ Image not found: {fname}")
else:
    st.warning("⚠️ Upload and process videos before searching.")
