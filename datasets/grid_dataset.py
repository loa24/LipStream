import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from utils.tokenizer import Tokenizer


class GridDataset(Dataset):
    def __init__(self, data_dir, vocab_file, img_size=112, split_file=None):
        self.data_dir = Path(data_dir)
        self.vocab_file = Path(vocab_file)
        self.img_size = img_size
        self.split_file = Path(split_file) if split_file else None

        self.tokenizer = Tokenizer(self.vocab_file)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.samples = self._load_samples()

    def _load_samples(self):
        sample_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if self.split_file is not None:
            with open(self.split_file, "r") as f:
                allowed = set(line.strip() for line in f if line.strip())

            sample_dirs = [d for d in sample_dirs if d.name in allowed]

        valid_samples = []

        for sample_dir in sample_dirs:
            frames_dir = sample_dir / "mouth_frames"
            transcript_path = sample_dir / "transcript.txt"

            if not frames_dir.exists():
                continue
            if not transcript_path.exists():
                continue

            frame_files = sorted(frames_dir.glob("*.jpg"))
            if len(frame_files) == 0:
                continue

            valid_samples.append(sample_dir)

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        frames_dir = sample_dir / "mouth_frames"
        transcript_path = sample_dir / "transcript.txt"

        frame_files = sorted(frames_dir.glob("*.jpg"))

        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        tokens = self.tokenizer.encode(transcript)
        tokens = torch.tensor(tokens, dtype=torch.long)

        return frames, tokens, transcript

    @staticmethod
    def collate_fn(batch):
        frames_list, tokens_list, transcripts = zip(*batch)

        max_frames = max(f.shape[0] for f in frames_list)
        frames_padded = []

        for frames in frames_list:
            if frames.shape[0] < max_frames:
                pad_size = max_frames - frames.shape[0]
                padding = torch.zeros(
                    pad_size,
                    frames.shape[1],
                    frames.shape[2],
                    frames.shape[3]
                )
                frames = torch.cat([frames, padding], dim=0)

            frames_padded.append(frames)

        frames_batch = torch.stack(frames_padded)

        max_tokens = max(t.shape[0] for t in tokens_list)
        tokens_padded = []

        for tokens in tokens_list:
            if tokens.shape[0] < max_tokens:
                pad_size = max_tokens - tokens.shape[0]
                padding = torch.full((pad_size,), 2, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=0)

            tokens_padded.append(tokens)

        tokens_batch = torch.stack(tokens_padded)

        frame_lengths = torch.tensor([f.shape[0] for f in frames_list], dtype=torch.long)
        target_lengths = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.long)

        return {
            "frames": frames_batch,
            "tokens": tokens_batch,
            "frame_lengths": frame_lengths,
            "target_lengths": target_lengths,
            "transcripts": transcripts
        }
