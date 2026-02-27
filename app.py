# ========================= # SINGLE-CELL TRAFFIC VIDEO AI APP # =========================
import streamlit as st
import os, cv2, tempfile, shutil, uuid
import torch
import numpy as np
from ultralytics import YOLO
from collections import Counter
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="🚦 Traffic Analysis Video AI", layout="wide")
st.title("🚦 Traffic Analysis (Video AI)")

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES     = 200     # cap for GPU safety
TARGET_FPS     = 2       # frame sampling rate
CONF_THRES     = 0.3     # detection confidence threshold
MAX_VIOLATIONS = 20      # cap stored violations to avoid memory bloat

VEHICLE_CLASSES  = {"car", "truck", "bus", "motorcycle"}

# -------------------------
# LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_yolo():
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
    return model

yolo = load_yolo()

# -------------------------
# UTILS
# -------------------------
def make_session_dir() -> str:
    """Create a unique per-session temp directory to avoid cross-user collisions."""
    session_id = st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    path = os.path.join(tempfile.gettempdir(), f"traffic_{session_id}")
    os.makedirs(path, exist_ok=True)
    return path


def extract_frames(video_path: str, frames_dir: str) -> tuple[list[str], float]:
    """
    Sample frames from a video at ~TARGET_FPS.

    Returns:
        (list of saved frame paths, actual video FPS detected)
    """
    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)

    # Warn if FPS detection is unreliable, but never silently default
    if not raw_fps or raw_fps <= 0:
        st.warning("Could not detect video FPS — defaulting to 30. Results may vary.")
        raw_fps = 30.0

    interval = max(1, int(raw_fps // TARGET_FPS))
    frames, idx, saved = [], 0, 0

    while cap.isOpened() and saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            saved += 1
        idx += 1

    cap.release()
    return frames, raw_fps


def detect_objects(img_path: str) -> list[dict]:
    """Run YOLO on a single frame; raises on error instead of silently returning []."""
    res = yolo(img_path, conf=CONF_THRES, verbose=False)[0]
    return [
        {"label": res.names[int(b.cls[0])], "conf": float(b.conf[0])}
        for b in res.boxes
    ]

# -------------------------
# TRAFFIC LOGIC
# -------------------------
def congestion_score(dets: list[dict]) -> int:
    """Count vehicles in a single frame's detections."""
    return sum(1 for d in dets if d["label"] in VEHICLE_CLASSES)


def triple_riding(dets: list[dict]) -> bool:
    """
    Flag when 3+ persons share a single motorcycle in the frame.
    Note: YOLO does not detect helmets (not a COCO class), so helmet
    detection would require a custom-trained model. That check has been
    removed from this version to avoid false positives.
    """
    persons = sum(1 for d in dets if d["label"] == "person")
    bikes   = sum(1 for d in dets if d["label"] == "motorcycle")
    return bikes == 1 and persons >= 3


def wrong_way_heuristic(prev_boxes: list, curr_boxes: list) -> bool:
    """
    Very lightweight heuristic: if vehicle centroid X moves strongly
    right-to-left across the majority of detections, flag as suspicious.
    Requires at least 3 matched vehicles.
    """
    if len(prev_boxes) < 3 or len(curr_boxes) < 3:
        return False
    prev_xs = sorted([b["cx"] for b in prev_boxes])
    curr_xs = sorted([b["cx"] for b in curr_boxes])
    deltas = [c - p for p, c in zip(prev_xs, curr_xs)]
    return len(deltas) >= 3 and all(d < -20 for d in deltas)

# -------------------------
# UI: VIDEO UPLOAD
# -------------------------
video_file = st.file_uploader("Upload traffic video (≤30 s recommended)", type=["mp4", "avi", "mov"])

if video_file:
    frames_dir = make_session_dir()

    # Write upload to a named temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=frames_dir) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("Extracting frames…")
    try:
        frames, detected_fps = extract_frames(video_path, frames_dir)
    except Exception as e:
        st.error(f"Frame extraction failed: {e}")
        st.stop()

    if not frames:
        st.error("No frames could be extracted from this video. Is the file valid?")
        st.stop()

    st.success(f"Extracted {len(frames)} frames at ~{TARGET_FPS} fps (source: {detected_fps:.1f} fps)")

    # -------------------------
    # PROCESS FRAMES
    # -------------------------
    congestion_data: list[int] = []
    violations: list[tuple[str, str]] = []   # (frame_path, violation_type)
    label_counter: Counter = Counter()
    errors: list[str] = []

    st.subheader("🔍 Processing Frames")
    prog = st.progress(0)

    for i, frame_path in enumerate(frames):
        try:
            dets = detect_objects(frame_path)
        except Exception as e:
            errors.append(f"Frame {i}: {e}")
            congestion_data.append(0)
            prog.progress((i + 1) / len(frames))
            continue

        for d in dets:
            label_counter[d["label"]] += 1

        congestion_data.append(congestion_score(dets))

        if triple_riding(dets) and len(violations) < MAX_VIOLATIONS:
            violations.append((frame_path, "Triple Riding"))

        prog.progress((i + 1) / len(frames))

    if errors:
        with st.expander(f"⚠️ {len(errors)} frame(s) failed detection"):
            st.write("\n".join(errors[:20]))

    # -------------------------
    # RESULTS
    # -------------------------
    st.markdown("---")
    st.header("📊 Traffic Analytics Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Frames",      len(frames))
    col2.metric("Avg Congestion",    round(float(np.mean(congestion_data)), 2))
    col3.metric("Peak Congestion",   int(np.max(congestion_data)))
    col4.metric("Violations Detected", len(violations))

    st.subheader("🚗 Object Frequency")
    if label_counter:
        # Show only traffic-relevant classes first, then others
        traffic_labels = {k: v for k, v in label_counter.items() if k in VEHICLE_CLASSES | {"person"}}
        other_labels   = {k: v for k, v in label_counter.items() if k not in traffic_labels}
        for k, v in sorted(traffic_labels.items(), key=lambda x: -x[1]):
            st.write(f"- **{k}** : {v}")
        if other_labels:
            with st.expander("Other detected objects"):
                for k, v in sorted(other_labels.items(), key=lambda x: -x[1]):
                    st.write(f"- {k} : {v}")
    else:
        st.write("No objects detected.")

    st.subheader("🚨 Violations Preview")
    if violations:
        cols = st.columns(min(3, len(violations[:6])))
        for idx, (frame_path, vtype) in enumerate(violations[:6]):
            cols[idx % 3].image(frame_path, caption=vtype, use_container_width=True)
        if len(violations) > 6:
            st.caption(f"…and {len(violations) - 6} more violation(s) not shown.")
    else:
        st.success("No violations detected.")

    st.subheader("📈 Congestion Over Time")
    st.line_chart(congestion_data)

    # -------------------------
    # CLEANUP
    # -------------------------
    try:
        shutil.rmtree(frames_dir)
        # Reset session dir so next upload gets a fresh one
        del st.session_state["session_id"]
    except Exception:
        pass  # non-fatal

else:
    st.info("Upload a traffic video to begin analysis.")
    st.caption(f"Running on: **{DEVICE.upper()}** | Max frames: {MAX_FRAMES} | Sampling: {TARGET_FPS} fps")
