import random
import shutil
from pathlib import Path

DATA_DIR = Path("data/processed/lipread/train")
OUTPUT_DIR = Path("data/debug/mouth_check_samples")

NUM_SAMPLES = 100
00
FRAMES_PER_SAMPLE = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get samples that already have mouth_frames
samples = [
    d for d in DATA_DIR.iterdir()
    if d.is_dir() and (d / "mouth_frames").exists()
]

print(f"Found {len(samples)} samples with mouth_frames")

selected_samples = random.sample(samples, min(NUM_SAMPLES, len(samples)))

for sample_dir in selected_samples:
    mouth_dir = sample_dir / "mouth_frames"
    frame_files = sorted(mouth_dir.glob("*.jpg"))

    if not frame_files:
        continue

    selected_frames = random.sample(
        frame_files,
        min(FRAMES_PER_SAMPLE, len(frame_files))
    )

    for frame_path in selected_frames:
        output_name = f"{sample_dir.name}_{frame_path.name}"
        output_path = OUTPUT_DIR / output_name
        shutil.copy(frame_path, output_path)

print(f"Copied review images to: {OUTPUT_DIR}")
print("Open this folder and inspect the crops.")
