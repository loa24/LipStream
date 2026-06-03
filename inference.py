import subprocess
import base64
import os
import uuid
import math
import tempfile
from pathlib import Path

import time
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.lipreading_model import LipReadingModel
from utils.tokenizer import Tokenizer

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = Path("checkpoints/lipreading_epoch_037.pth")
VOCAB_FILE  = Path("data/vocab.txt")
YOLO_MODEL  = "yolov8n.pt"
IMG_SIZE    = 112
NUM_CLASSES = 55
HIDDEN_SIZE = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app, origins="*")

app.logger.setLevel(logging.INFO)

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading model on {DEVICE}...")
tokenizer = Tokenizer(VOCAB_FILE)

model = LipReadingModel(
    num_classes=NUM_CLASSES,
    hidden_size=HIDDEN_SIZE,
    dropout_rate=0.3,
    freeze_cnn=False,
).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded.")

yolo = YOLO(YOLO_MODEL)

# ── preprocessing ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def extract_mouth_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    mouth_frames = []
    for frame in frames:
        results = yolo(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            h, w = frame.shape[:2]
            crop = frame[h//2:, w//4: 3*w//4]
        else:
            x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy().astype(int)
            face_h = y2 - y1
            mouth_y1 = y1 + int(face_h * 0.65)
            mouth_y2 = y2
            crop = frame[mouth_y1:mouth_y2, x1:x2]

        if crop.size == 0:
            crop = frame

        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mouth_frames.append(img)

    return mouth_frames


def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int) -> list:
    indices = logits.argmax(dim=-1).tolist()
    decoded = []
    prev = None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            decoded.append(idx)
        prev = idx
    return decoded


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(video_path: str) -> dict:
    mouth_imgs = extract_mouth_frames(video_path)

    if not mouth_imgs:
        return {"error": "No frames extracted from video"}

    frame_tensors = [transform(img) for img in mouth_imgs]
    frames = torch.stack(frame_tensors).unsqueeze(0).to(DEVICE)  # (1, T, 3, H, W)

    with torch.no_grad():
        logits = model(frames)          # (T, 1, num_classes)

    logits = logits.squeeze(1)          # (T, num_classes)
    probs  = torch.softmax(logits, dim=-1)

    token_ids  = ctc_greedy_decode(logits, tokenizer.blank_idx)
    transcript = tokenizer.decode(token_ids)

    confidence = float(probs.max(dim=-1).values.mean().item())

    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = round(frame_count / fps, 1)

    words     = [w for w in transcript.split() if w]
    sentences = max(1, math.ceil(len(words) / 6))
    word_duration = duration / len(words) if words else duration
    
    total_frames = frames.shape[1]
    

    timed_words = []
    for i, word in enumerate(words):
        start_sec = round(i * word_duration, 1)
        timed_words.append({"time": start_sec, "word": word})

    with open(video_path, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode("utf-8")

    return {
        "transcript": transcript,
        "confidence": round(confidence, 4),
        "duration":   duration,
        "sentences":  sentences,
        "words":      len(words),
        "quality":    "High" if confidence >= 0.75 else "Medium" if confidence >= 0.5 else "Low",
        "timed_words": timed_words,
        "video_b64":  video_b64,
    }

# ── 

def convert_to_mp4(input_path: str) -> str:
    output_path = input_path.replace(Path(input_path).suffix, ".mp4")
    subprocess.run([
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac",
        output_path, "-y", "-loglevel", "error"
    ], check=True)
    return output_path

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()  

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    suffix     = Path(video_file.filename).suffix or ".mp4"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    video_file.save(tmp_path)
    tmp.close()

    try:
        if Path(tmp_path).suffix.lower() != ".mp4":
            converted_path = convert_to_mp4(tmp_path)
            os.unlink(tmp_path)
            tmp_path = converted_path

        result = run_inference(tmp_path)
    except Exception as e:
        end_time = time.time()  
        e2e_latency_ms = (end_time - start_time) * 1000.0
        app.logger.info(f"E2E latency (error): {e2e_latency_ms:.2f} ms")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    end_time = time.time()
    e2e_latency_ms = (end_time - start_time) * 1000.0
    app.logger.info(f"E2E latency: {e2e_latency_ms:.2f} ms")

    result["e2e_latency_ms"] = round(e2e_latency_ms, 2)

    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)