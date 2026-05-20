import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json

from datasets.grid_dataset import GridDataset
from models.lipreading_model import LipReadingModel


def flatten_targets(tokens, target_lengths):
    """
    Convert padded token sequences to flattened 1D target tensor for CTC loss.
    
    Args:
        tokens: (B, L_max) padded token indices
        target_lengths: (B,) actual token counts per sample
    
    Returns:
        targets: (sum of target_lengths,) flattened tensor
    """
    targets = []
    for i in range(tokens.shape[0]):
        targets.append(tokens[i, :target_lengths[i].item()])
    return torch.cat(targets)


def train_one_epoch(model, loader, optimizer, ctc_loss, device, epoch, log_interval=20):
    """
    Train for one epoch.
    
    Args:
        model: LipReadingModel
        loader: training DataLoader
        optimizer: Adam optimizer
        ctc_loss: CTCLoss criterion
        device: cuda or cpu
        epoch: current epoch number
        log_interval: print loss every N batches
    
    Returns:
        avg_loss: average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        frames = batch["frames"].to(device)              # (B, T_max, 3, 112, 112)
        tokens = batch["tokens"].to(device)              # (B, L_max)
        frame_lengths = batch["frame_lengths"].to(device)  # (B,) actual frame counts
        target_lengths = batch["target_lengths"].to(device)  # (B,) actual token counts

        # Flatten targets for CTC (remove padding)
        targets = flatten_targets(tokens, target_lengths).to(device)  # (sum_of_lengths,)

        # Forward pass
        outputs = model(frames)  # (T_max, B, num_classes)
        log_probs = torch.log_softmax(outputs, dim=2)  # Required by CTC

        # Compute loss
        loss = ctc_loss(log_probs, targets, frame_lengths, target_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for LSTM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if batch_idx % log_interval == 0:
            batch_loss = loss.item()
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {batch_loss:.4f}")

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch {epoch} training completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f}")
    
    return avg_loss



def save_checkpoint(checkpoint_dir, model, optimizer, epoch, train_loss, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        checkpoint_dir: directory to save checkpoints
        model: the model
        optimizer: the optimizer
        epoch: current epoch
        train_loss: training loss
        is_best: whether this is the best checkpoint
    
    Returns:
        checkpoint_path: path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"lipreading_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "lipreading_best.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path}")
        return best_path
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load checkpoint to resume training.
    
    Args:
        checkpoint_path: path to checkpoint
        model: the model
        optimizer: the optimizer (optional)
    
    Returns:
        epoch: starting epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch 


def create_data_loaders(batch_size, num_workers=0):
    """
    Create train and validation data loaders.
    
    Args:
        batch_size: batch size
        num_workers: number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Training dataset
    train_dataset = GridDataset(
        "data/processed/lipread/train",
        "data/vocab.txt",
        img_size=112,
       
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=GridDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    
    return train_loader


def main():
    """
    Main training loop with validation and checkpointing.
    """
    # ========== Configuration ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    epochs = 3
    lr = 1e-4
    num_classes = 55
    num_workers = 0 
    
    # Paths
    checkpoint_dir = Path("checkpoints")
    log_file = checkpoint_dir / "training_log.json"
    resume_checkpoint = None  # Set to path if resuming from checkpoint
    
    # ========== Data Loading ==========
    print("\nLoading datasets...")
    train_loader = create_data_loaders(batch_size, num_workers)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # ========== Model Initialization ==========
    print("\nInitializing model...")
    model = LipReadingModel(
        num_classes=num_classes,
        hidden_size=256,
        dropout_rate=0.3,
        freeze_cnn=True  
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========== Optimizer & Loss ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Optional: learning rate scheduler (slows down learning over time)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # ========== Resume from checkpoint ==========
    start_epoch = 1
    
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        start_epoch= load_checkpoint(
            resume_checkpoint, model, optimizer
        )
        start_epoch += 1  # Start from next epoch
    
    # ========== Training Loop ==========
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    training_history = []
    
    print(f"\nStarting training from epoch {start_epoch} to {epochs}...\n")
    
    for epoch in range(start_epoch, epochs + 1):
        print("=" * 60)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 60)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ctc_loss, device, epoch
        )
        
        
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        save_checkpoint(
              checkpoint_dir,
              model, optimizer, epoch,
              train_loss
        )
        
       
        # Log to history
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": current_lr,
        }
        training_history.append(history_entry)
        
        # Save training history
        with open(log_file, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print()
    
    # ========== Training Complete ==========
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training log: {log_file}")


if __name__ == "__main__":
    main()

