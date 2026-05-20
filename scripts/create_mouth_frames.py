"""
Mouth region extraction pipeline for lip-reading.
Processes video frames and extracts mouth regions using YOLO detection.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import gc
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for mouth extraction."""
    
    DATA_DIR = Path("data/processed/lipread/train")
    LOG_DIR = Path("logs")
    
    # Mouth region as proportion of face box
    MOUTH_REGION = {
    'x_min': 0.20,
    'x_max': 0.80,
    'y_min': 0.55,
    'y_max': 0.92

    }
    
    # Fallback region if YOLO fails
    FALLBACK_REGION = {
    'x_min': 0.20,
    'x_max': 0.80,
    'y_min': 0.50,
    'y_max': 0.95

    }
    
    OUTPUT_SIZE = (112, 112)
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONFIDENCE = 0.5
    RESIZE_INTERPOLATION = cv2.INTER_LINEAR
    LOG_INTERVAL = 100

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_dir: Path = Config.LOG_DIR) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# STATISTICS
# ============================================================================

class ExtractionStats:
    """Track extraction statistics."""
    
    def __init__(self):
        self.total_samples = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.total_frames = 0
        self.successful_frames = 0
        self.fallback_frames = 0
        self.failed_frames = 0
        self.errors = []
    
    def add_sample_success(self, frame_count: int, fallback_count: int):
        self.successful_samples += 1
        self.total_frames += frame_count
        self.successful_frames += frame_count - fallback_count
        self.fallback_frames += fallback_count
    
    def add_sample_failure(self, error: str):
        self.failed_samples += 1
        self.errors.append(error)
    
    def report(self) -> str:
        """Generate statistics report."""
        report = f"""
{'='*60}
EXTRACTION STATISTICS
{'='*60}
Samples:
  Total:       {self.total_samples}
  Successful:  {self.successful_samples}
  Failed:      {self.failed_samples}

Frames:
  Total:       {self.total_frames}
  Successful:  {self.successful_frames}
  Fallback:    {self.fallback_frames}
  Failed:      {self.failed_frames}

Success Rate (samples): {100*self.successful_samples/max(1,self.total_samples):.1f}%
Success Rate (frames):  {100*self.successful_frames/max(1,self.total_frames):.1f}%

Errors: {len(self.errors)}
{'='*60}
"""
        return report

stats = ExtractionStats()

# ============================================================================
# YOLO MODEL
# ============================================================================

def load_yolo_model(model_name: str = Config.YOLO_MODEL) -> Optional[YOLO]:
    """Load YOLO model with error handling."""
    try:
        logger.info(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name)
        logger.info(f" YOLO model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f" Model not found: {model_name}")
        logger.error("Download with: yolo detect download model=yolov8n.pt")
        return None
    except Exception as e:
        logger.error(f" Error loading YOLO: {e}")
        return None

# ============================================================================
# MOUTH EXTRACTION
# ============================================================================

def crop_mouth_from_frame(
    frame: np.ndarray,
    model: YOLO,
    use_fallback: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Extract mouth region from frame.
    
    Args:
        frame: Input image frame
        model: YOLO model for detection
        use_fallback: Whether to use fallback on detection failure
    
    Returns:
        (mouth_crop, used_fallback): Cropped mouth region and fallback flag
    """
    h, w = frame.shape[:2]
    
    # Get fallback coordinates
    fx1 = int(Config.FALLBACK_REGION['x_min'] * w)
    fx2 = int(Config.FALLBACK_REGION['x_max'] * w)
    fy1 = int(Config.FALLBACK_REGION['y_min'] * h)
    fy2 = int(Config.FALLBACK_REGION['y_max'] * h)
    
    # Try YOLO detection
    try:
        results = model(frame, verbose=False, conf=Config.YOLO_CONFIDENCE)
        
        if len(results[0].boxes) == 0:
            if use_fallback:
                return frame[fy1:fy2, fx1:fx2], True
            return np.array([]), False
        
        # Get largest detected box
        boxes = results[0].boxes.xyxy.cpu().numpy()
        largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        
        x1, y1, x2, y2 = map(int, largest_box)
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Calculate mouth region
        mx1 = x1 + int(Config.MOUTH_REGION['x_min'] * face_w)
        mx2 = x1 + int(Config.MOUTH_REGION['x_max'] * face_w)
        my1 = y1 + int(Config.MOUTH_REGION['y_min'] * face_h)
        my2 = y1 + int(Config.MOUTH_REGION['y_max'] * face_h)
        
        # Clip to boundaries
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(w, mx2), min(h, my2)
        
        mouth = frame[my1:my2, mx1:mx2]
        
        if mouth.size == 0:
            if use_fallback:
                return frame[fy1:fy2, fx1:fx2], True
            return np.array([]), False
        
        return mouth, False
    
    except Exception as e:
        logger.warning(f"YOLO detection failed: {e}")
        if use_fallback:
            return frame[fy1:fy2, fx1:fx2], True
        return np.array([]), False

# ============================================================================
# SAMPLE PROCESSING
# ============================================================================

def validate_data_structure(data_dir: Path) -> bool:
    """Validate data directory structure."""
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    samples = [d for d in data_dir.iterdir() if d.is_dir()]
    if not samples:
        logger.error(f"No sample directories found in {data_dir}")
        return False
    
    logger.info(f" Found {len(samples)} sample directories")
    return True

def process_sample(
    sample_dir: Path,
    model: YOLO
) -> Dict:
    """
    Process all frames in a sample.
    
    Returns:
        dict: Processing results with status and details
    """
    try:
        frames_dir = sample_dir / "frames"
        mouth_dir = sample_dir / "mouth_frames"
        
        if not frames_dir.exists():
            return {
                'status': 'failed',
                'processed': 0,
                'error': f"Frames directory not found"
            }
        
        mouth_dir.mkdir(exist_ok=True)
        frame_files = sorted([
             f for f in frames_dir.iterdir()
             if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        if not frame_files:
            return {
                'status': 'failed',
                'processed': 0,
                'error': f"No .jpg files found"
            }
        
        processed = 0
        skipped = 0
        failed = 0
        fallback_count = 0
        
        for frame_path in frame_files:
            output_path = mouth_dir / frame_path.name
            
            # Skip if already processed
            if output_path.exists():
                skipped += 1
                continue
            
            try:
                # Read frame
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    failed += 1
                    continue
                
                # Extract mouth
                mouth, used_fallback = crop_mouth_from_frame(frame, model)
                if mouth.size == 0:
                    failed += 1
                    continue
                
                if used_fallback:
                    fallback_count += 1
                
                # Resize and save
                mouth = cv2.resize(mouth, Config.OUTPUT_SIZE, 
                                 interpolation=Config.RESIZE_INTERPOLATION)
                cv2.imwrite(str(output_path), mouth)
                processed += 1
            
            except Exception as e:
                logger.debug(f"Failed to process {frame_path.name}: {e}")
                failed += 1
        
        return {
            'status': 'success',
            'processed': processed,
            'skipped': skipped,
            'failed': failed,
            'fallback': fallback_count,
            'error': None
        }
    
    except Exception as e:
        logger.error(f"Sample processing failed: {e}")
        return {
            'status': 'failed',
            'processed': 0,
            'error': str(e)
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Starting mouth extraction pipeline")
    logger.info("="*60)
    
    # Validate
    if not validate_data_structure(Config.DATA_DIR):
        logger.error("Data validation failed. Exiting.")
        return
    
    # Load model
    model = load_yolo_model()
    if model is None:
        logger.error("Failed to load YOLO model. Exiting.")
        return
    
    # Get samples
    samples = sorted([d for d in Config.DATA_DIR.iterdir() if d.is_dir()])
    stats.total_samples = len(samples)
    
    logger.info(f"Processing {len(samples)} samples...")
    
    try:
        for i, sample_dir in enumerate(samples, start=1):
            try:
                result = process_sample(sample_dir, model)
                
                if result['status'] == 'success':
                    stats.add_sample_success(
                        result.get('processed', 0) + result.get('failed', 0),
                        result.get('fallback', 0)
                    )
                    logger.debug(f" {sample_dir.name}: {result['processed']} frames")
                else:
                    stats.add_sample_failure(result.get('error', 'Unknown error'))
                    logger.warning(f"{sample_dir.name}: {result['error']}")
            
            except Exception as e:
                stats.add_sample_failure(str(e))
                logger.error(f"{sample_dir.name}: {e}")
            
            finally:
                # Periodic cleanup
                if i % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"Progress: {i}/{len(samples)} samples")
    
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(stats.report())
        logger.info("Pipeline completed")

if __name__ == "__main__":
    main()

