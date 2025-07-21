import streamlit as st
import os
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# ==============================
# CONFIG
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="Image Search (YOLO + CLIP)", layout="wide")
st.title("🖼️🔎 Image Search with YOLOv8 + CLIP")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # lightest YOLOv8 model
yolo_model = load_yolo()

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer
clip_model, clip_preprocess, tokenizer = load_clip()

# ==============================
# DATA STORAGE
# ==============================
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

def save_metadata(image_files, objects, embeddings):
    # save filenames
    with open("metadata.txt", "w") as f:
        for fname in image_files:
            f.write(f"{fname}\n")
    # save detected objects
    with open("objects.json", "w") as f:
        json.dump(objects, f, indent=2)
    # save embeddings
    np.save("embeddings.npy", np.vstack(embeddings))

# ==============================
# HELPERS
# ==============================
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

# ==============================
# IMAGE UPLOAD
# ==============================
uploaded_images = st.file_uploader(
    "📤 Upload image(s) to index",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_images:
    with st.spinner("⏳ Processing uploaded images..."):
        all_files = []
        all_objects = {}
        all_embeddings = []

        for img_file in uploaded_images:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1]) as tmp:
                tmp.write(img_file.read())
                tmp_path = tmp.name

            # save image permanently in frames/
            base_name = os.path.basename(tmp_path)
            dst_path = os.path.join(FRAME_DIR, base_name)
            os.replace(tmp_path, dst_path)

            objs = detect_objects(dst_path)
            emb = encode_image(dst_path)

            all_files.append(base_name)
            all_objects[base_name] = objs
            all_embeddings.append(emb)

        if all_embeddings:
            save_metadata(all_files, all_objects, all_embeddings)
            st.success("✅ Images processed and indexed successfully!")
        else:
            st.error("⚠️ No embeddings generated.")

# ==============================
# SEARCH
# ==============================
st.markdown("---")
st.header("🔎 Search in indexed images")

if os.path.exists("embeddings.npy"):
    query = st.text_input("Enter a text description to search:")
    if query:
        # load metadata
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))
        embeddings = np.load("embeddings.npy")

        # encode query
        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        query_vec = tfeat.cpu().numpy()

        # cosine similarity
        scores = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:5]

        st.subheader(f"Top matches for **'{query}'**:")
        for idx in top_indices:
            fname = metadata[idx]
            score = scores[idx]
            img_path = os.path.join(FRAME_DIR, fname)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{fname} (score {score:.3f})", use_container_width=True)
                st.write("Detected objects:")
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")
            else:
                st.write(f"❌ Missing image file: {fname}")
else:
    st.warning("⚠️ Upload and index images first.")
