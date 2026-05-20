import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class GridDataset(Dataset):
    """
    GRID Dataset loader for visual speech recognition.
    
    Dataset structure:
    data_dir/
    ├── sample_1/
    │   ├── frames/
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   └── ...
    │   └── transcript.txt
    ├── sample_2/
    │   ├── frames/
    │   └── transcript.txt
    └── ...
    """

    def __init__(
        self,
        data_dir: str,
        vocab_file: str,
        img_size: int = 224,
        min_frames: int = 25,
        max_frames: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Args:
            data_dir: Path to dataset root directory
            vocab_file: Path to vocabulary file
            img_size: Image size for resizing (default: 224 for ResNet-50)
            min_frames: Minimum frames required per sample
            max_frames: Maximum frames to load (None = unlimited)
            verbose: Enable logging
        """
        self.data_dir = Path(data_dir)
        self.vocab_file = Path(vocab_file)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.verbose = verbose

        #  Validate input paths
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")

        # Initialize tokenizer
        self.tokenizer = Tokenizer(self.vocab_file, verbose=verbose)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

        #  Load and validate samples
        self.samples = self._load_samples()

        if self.verbose:
            print(f"✓ Loaded {len(self.samples)} samples from {self.data_dir}")

    def _load_samples(self) -> list:
        """
        Load and validate sample directories.
        
        Returns:
            List of valid sample directory paths
        """
        samples = []
        
        #  Filter only directories (not files)
        sample_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        for sample_dir in sample_dirs:
            try:
                # Check required structure
                frames_dir = sample_dir / "mouth_frames"
                transcript_path = sample_dir / "transcript.txt"

                if not frames_dir.exists():
                    logger.warning(f"Frames directory missing: {frames_dir}")
                    continue

                if not transcript_path.exists():
                    logger.warning(f"Transcript missing: {transcript_path}")
                    continue

                # Check for frame files
                frame_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
                if len(frame_files) < self.min_frames:
                    logger.warning(
                        f"Sample {sample_dir.name}: "
                        f"Only {len(frame_files)} frames (min: {self.min_frames})"
                    )
                    continue

                samples.append(sample_dir)

            except Exception as e:
                logger.warning(f"Error validating {sample_dir}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get sample by index.
        
        Returns:
            Tuple of (frames, tokens, transcript)
            - frames: (T, 3, 224, 224) where T is number of frames
            - tokens: (L,) where L is number of tokens
            - transcript: original text
        """
        sample_dir = self.samples[idx]
        
        try:
            #  Load frames with error handling
            frames = self._load_frames(sample_dir)
            
            #  Load and tokenize transcript with error handling
            transcript = self._load_transcript(sample_dir)
            tokens = self.tokenizer.encode(
                transcript,
                handle_unknown="unk"  #  Use default mode
            )
            tokens = torch.tensor(tokens, dtype=torch.long)

            return frames, tokens, transcript

        except Exception as e:
            logger.error(f"Error loading sample {sample_dir.name}: {e}")
            #  Return dummy data or skip sample gracefully
            # (or raise exception for DataLoader to skip)
            raise

    def _load_frames(self, sample_dir: Path) -> torch.Tensor:
        """
        Load and preprocess frames.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Tensor of shape (T, 3, 224, 224)
        """
        frames_dir = sample_dir / "mouth_frames"
        
        #  Get image files with proper extension filtering
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])

        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")

        #  Limit frames if max_frames specified
        if self.max_frames is not None:
            frame_files = frame_files[:self.max_frames]

        frames = []
        failed_frames = []

        for frame_path in frame_files:
            try:
                # Handle corrupted images
                img = Image.open(frame_path).convert("RGB")
                img = self.transform(img)
                frames.append(img)
            except Exception as e:
                failed_frames.append((frame_path.name, str(e)))
                logger.warning(f"Failed to load {frame_path.name}: {e}")

        if not frames:
            raise ValueError(
                f"All frames failed to load in {sample_dir.name}: "
                f"{failed_frames}"
            )

        if failed_frames and self.verbose:
            logger.warning(
                f"Sample {sample_dir.name}: "
                f"Failed to load {len(failed_frames)}/{len(frame_files)} frames"
            )

        #  Stack frames: (T, 3, 224, 224)
        frames = torch.stack(frames)

        return frames

    def _load_transcript(self, sample_dir: Path) -> str:
        """
        Load and validate transcript.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Transcript text
        """
        transcript_path = sample_dir / "transcript.txt"

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

            if not transcript:
                raise ValueError(f"Empty transcript in {sample_dir.name}")

            return transcript

        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {transcript_path}: {e}")
            raise
   
    @staticmethod
    def collate_fn(batch: list) -> dict:
        frames_list, tokens_list, transcripts = zip(*batch)

        # Padding frames
        frame_lengths = torch.tensor([f.shape[0] for f in frames_list], dtype=torch.long)
        max_frames = frame_lengths.max().item()
    
        frames_padded = []
        for frames in frames_list:
            if frames.shape[0] < max_frames:
                pad_size = max_frames - frames.shape[0]
                frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_size))
            frames_padded.append(frames)
    
        frames_batch = torch.stack(frames_padded)

       # Padding tokens
        target_lengths = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.long)
        max_tokens = target_lengths.max().item()
    
        tokens_padded = []
        for tokens in tokens_list:
            if tokens.shape[0] < max_tokens:
                 pad_size = max_tokens - tokens.shape[0]
                 tokens = torch.nn.functional.pad(tokens, (0, pad_size), value=2)
            tokens_padded.append(tokens)
    
        tokens_batch = torch.stack(tokens_padded)
    
        return {
             "frames": frames_batch,
             "tokens": tokens_batch,
             "frame_lengths": frame_lengths,
             "target_lengths": target_lengths,
             "transcripts": transcripts
           }

