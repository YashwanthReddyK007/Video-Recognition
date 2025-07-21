import streamlit as st
import os
import tempfile
import json
import torch
import numpy as np
from PIL import Image
from collections import Counter
import faiss
import open_clip
from ultralytics import YOLO
import shutil

# =======================
# SETTINGS
# =======================
FRAMES_FOLDER = "frames"
os.makedirs(FRAMES_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SCORE_THRESHOLD = 0.25  # 👈 minimum similarity score to consider a valid match

st.set_page_config(page_title="📷 Image Search with YOLO + CLIP", layout="wide")
st.title("📷🔎 Image Search with YOLOv8 + CLIP")

# =======================
# LOAD MODELS
# =======================
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = YOLO("yolov8n.pt")
yolo_model = st.session_state.yolo_model

@st.cache_resource
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(device).eval(), preprocess, tokenizer

clip_model, clip_preprocess, tokenizer = load_clip()

# =======================
# FUNCTIONS
# =======================
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
# IMAGE UPLOAD & PROCESS
# =======================
uploaded_images = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_images:
    all_filenames = []
    all_objects = {}
    all_embeddings = []

    for image_file in uploaded_images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as tmp:
            tmp.write(image_file.read())
            tmp_path = tmp.name

        fname = os.path.basename(tmp_path)
        dst_path = os.path.join(FRAMES_FOLDER, fname)
        shutil.move(tmp_path, dst_path)
        all_filenames.append(fname)

        objs = detect_objects(dst_path)
        all_objects[fname] = objs
        emb = encode_image(dst_path)
        all_embeddings.append(emb)

    with open("metadata.txt", "w") as f:
        for m in all_filenames:
            f.write(f"{m}\n")
    with open("objects.json", "w") as f:
        json.dump(all_objects, f, indent=2)

    if all_embeddings:
        dim = all_embeddings[0].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(all_embeddings))
        faiss.write_index(index, "image_index.faiss")
        st.success("✅ Images processed successfully! You can now search.")
    else:
        st.error("⚠️ No embeddings created.")

st.markdown("---")
st.header("🔎 Search Images")

if os.path.exists("image_index.faiss") and os.path.exists("metadata.txt"):
    query = st.text_input("Enter a description (e.g. 'car', 'dog'):")
    if query:
        index = faiss.read_index("image_index.faiss")
        metadata = [line.strip() for line in open("metadata.txt")]
        objects_map = json.load(open("objects.json"))

        tok = tokenizer([query])
        with torch.no_grad():
            tfeat = clip_model.encode_text(tok.to(device))
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        D, I = index.search(tfeat.cpu().numpy(), k=5)

        # filter results by threshold
        valid_results = [(idx, score) for idx, score in zip(I[0], D[0]) if score >= MIN_SCORE_THRESHOLD]

        if not valid_results:
            st.warning("⚠️ No results found for your query.")
        else:
            st.subheader(f"Top matches for **'{query}'**:")
            for idx, score in valid_results:
                fname = metadata[idx]
                img_path = os.path.join(FRAMES_FOLDER, fname)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"{fname} (score {score:.3f})", width=400)
                else:
                    st.write(f"❌ Image not found: {img_path}")

                st.write("Objects detected:")
                for obj in objects_map.get(fname, []):
                    st.write(f"- **{obj['label']}** ({obj['conf']:.2f})")

    st.markdown("---")
    st.header("📋 Summarize All Uploaded Images")
    if st.button("Summarize Objects"):
        objects_map = json.load(open("objects.json"))
        all_labels = [obj["label"] for objs in objects_map.values() for obj in objs]
        if not all_labels:
            st.warning("No objects detected yet.")
        else:
            counts = Counter(all_labels)
            st.write("✅ **Summary of objects detected:**")
            for label, count in counts.most_common():
                st.write(f"- **{label}** appears in {count} image(s)")
else:
    st.info("Upload and process images first.")
