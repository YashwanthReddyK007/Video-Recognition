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
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Video Search (YOLO + CLIP)", layout="wide")
st.title("🎥🔎 Video Search with YOLOv8 + CLIP")

# =======================
# LOAD MODELS
# =======================
@st.cache_resource
def load_yolo():
    # smallest YOLOv8n model
    return YOLO("yolov8n.pt")
yolo_model = load_yolo()

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer
clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# HELPERS
# =======================
def extract_frames(video_path, every_n=30):
    """Extract frames every N frames and save to folder."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            fname = f"{os.path.basename(video_path)}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            cv2.imwrite(fpath, frame)
            saved.append(fname)
        count += 1
    cap.release()
    return saved

def detect_objects(img_path):
    """Detect objects with YOLO and return labels/confidences."""
    results = yolo_model(img_path, conf=0.3, verbose=False)
    objs = []
    for box in results[0].boxes:
        objs.append({
            "label": results[0].names[int(box.cls[0])],
            "conf": float(box.conf[0])
        })
    return objs

def encode_image(img_path):
    """Generate CLIP embedding for an image."""
    img = Image.open(img_path).convert("RGB")
    tens = clip_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(tens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

# =======================
# VIDEO UPLOAD & PROCESS
# =======================
uploaded_videos = st.file_uploader(
    "Upload one or more videos",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True
)

if uploaded_videos:
    st.info("⏳ Processing uploaded videos...")
    all_frames = []
    all_objects = {}
    all_embeddings = []

    for video in uploaded_videos:
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
        st.success("✅ Processing complete! You can now search for frames.")
    else:
        st.error("⚠️ No frames processed!")

st.markdown("---")
st.header("🔎 Search in processed frames")

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
                st.image(img_path, caption=f"{fname} (score {score:.3f})", use_container_width=True)
            else:
                st.write(f"❌ Image file not found: {img_path}")
            st.write("Objects detected in this frame:")
            for obj in objects_map.get(fname, []):
                st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
else:
    st.warning("⚠️ Please upload and process videos first.")
