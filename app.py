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
from tqdm import tqdm

# =======================
# SETTINGS
# =======================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_clip():
    m, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return m.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-B-32")

yolo_model = load_yolo()
clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# Helper functions
# =======================
def extract_frames(video_path, every_n=30):
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
    results = yolo_model(img_path, conf=0.3, verbose=False)
    objs = []
    for box in results[0].boxes:
        objs.append({
            "label": results[0].names[int(box.cls[0])],
            "conf": float(box.conf[0])
        })
    return objs

def encode_image(img_path):
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

uploaded_videos = st.file_uploader("Upload video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_videos:
    st.info("Processing uploaded videos...")
    all_frames = []
    all_objects = {}
    all_embeddings = []

    for video in uploaded_videos:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=video.name) as tmp:
            tmp.write(video.read())
            tmp_path = tmp.name

        st.write(f"📥 Extracting frames from {video.name}...")
        frames = extract_frames(tmp_path, every_n=30)
        all_frames.extend(frames)

        for frame in tqdm(frames):
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
        st.success("✅ Processing complete! You can now search for objects.")
    else:
        st.error("No frames processed!")

st.markdown("---")
st.header("🔎 Search in processed videos")

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

        st.subheader(f"Top matches for '{query}':")
        for idx, score in zip(I[0], D[0]):
            fname = metadata[idx]
            st.image(os.path.join(FRAMES_FOLDER, fname), caption=f"{fname} (score {score:.3f})")
            st.write("Objects detected in this frame:")
            for obj in objects_map.get(fname, []):
                st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
else:
    st.warning("⚠️ Please upload and process videos first.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, YOLOv8, and CLIP")
