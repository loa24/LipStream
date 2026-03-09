import os
import cv2
import json
from pathlib import Path


RAW_GRID_DIR = Path("data/raw/GRID/grid1")
PROCESSED_DIR = Path("data/processed/lipread/train")


def read_transcript(align_path):
    words = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, _, word = parts
            if word.lower() != "sil":
                words.append(word)
    return " ".join(words)


def extract_frames(video_path, output_frames_dir):
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 1

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = output_frames_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_idx += 1

    cap.release()
    return fps


def process_one_sample(speaker_dir, video_file, sample_id):
    video_path = speaker_dir / video_file
    base_name = video_path.stem
    align_path = speaker_dir / "align" / f"{base_name}.align"

    if not align_path.exists():
        print(f"Missing align file for {video_path.name}")
        return

    sample_dir = PROCESSED_DIR / sample_id
    frames_dir = sample_dir / "frames"
    sample_dir.mkdir(parents=True, exist_ok=True)

    transcript = read_transcript(align_path)
    fps = extract_frames(video_path, frames_dir)

    with open(sample_dir / "transcript.txt", "w") as f:
        f.write(transcript)

    meta = {
        "dataset": "GRID",
        "speaker": speaker_dir.name,
        "language": "en",
        "fps": fps,
        "original_video": str(video_path)
    }

    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Processed {video_file} -> {sample_dir}")


def main():
    speaker_dir = RAW_GRID_DIR / "s1_processed"

    video_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith(".mpg")])

    # only process first 3 videos for testing
    for i, video_file in enumerate(video_files[:3], start=1):
        sample_id = f"sample_{i:06d}"
        process_one_sample(speaker_dir, video_file, sample_id)


if __name__ == "__main__":
    main()

