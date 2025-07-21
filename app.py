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
st.set_page_config(page_title="Video Search (YOLO + CLIP)", layout="wide")
st.title("🎥🔎 Video Search with YOLOv8 + CLIP")

# =======================
# LOAD MODELS
# =======================
# Do not cache YOLO (pickle issues on Streamlit Cloud)
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO("yolov8n.pt")
yolo_model = st.session_state.yolo_model

# Cache CLIP
@st.cache_resource
def load_clip():
    m, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return m.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# HELPERS
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
            if frame is None or frame.size == 0:
                print(f"[WARN] Empty frame at {count}")
                continue
            h, w, _ = frame.shape
            # Optional resize for efficiency
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            fname = f"{os.path.basename(video_path)}_{count}.jpg"
            fpath = os.path.join(FRAMES_FOLDER, fname)
            img_pil.save(fpath, format="JPEG")
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
# VIDEO UPLOAD
# =======================
uploaded_videos = st.file_uploader("📤 Upload video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True)

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
            emb = encode_image(fpath)
            all_objects[frame] = objs
            all_embeddings.append(emb)

    # Save metadata
    with open("metadata.txt", "w") as f:
        for m in all_frames:
            f.write(f"{m}\n")
    with open("objects.json", "w") as f:
        json.dump(all_objects, f, indent=2)

    # Build FAISS
    if all_embeddings:
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "video_index.faiss")
        st.success("✅ Processing complete! You can now search.")
    else:
        st.error("⚠️ No valid frames found! Try another video.")

# =======================
# SEARCH SECTION
# =======================
st.markdown("---")
st.header("🔎 Search in processed frames")

if os.path.exists("video_index.faiss") and os.path.exists("metadata.txt") and os.path.exists("objects.json"):
    query = st.text_input("Enter text to search:")
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
                st.write(f"❌ Missing file: {img_path}")
            st.write("Objects detected:")
            for obj in objects_map.get(fname, []):
                st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")

    st.markdown("---")
    st.header("📋 Summarize video")
    if st.button("Summarize objects"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected.")
        else:
            counts = Counter(all_labels)
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} frames")
else:
    st.warning("⚠️ Upload and process videos first.")
