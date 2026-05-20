import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.grid_dataset import GridDataset
from models.lipreading_model import LipReadingModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = GridDataset(
        "data/processed/lipread/train",
        "data/vocab.txt",
        img_size=112
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=GridDataset.collate_fn,
        num_workers=0
    )

    batch = next(iter(loader))

    frames = batch["frames"].to(device)  # (B, T_max, 3, 112, 112)
    tokens = batch["tokens"].to(device)  # (B, L_max)
    frame_lengths = batch["frame_lengths"].to(device)  #  Actual frame counts
    target_lengths = batch["target_lengths"].to(device)  #  Actual token counts

    print(f"Frames shape: {frames.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Frame lengths: {frame_lengths}")
    print(f"Target lengths: {target_lengths}")

    # Convert padded tokens to flat targets (remove padding)
    targets = []
    for i in range(tokens.shape[0]):
        length = target_lengths[i].item()
        # Extract only actual tokens (exclude PAD)
        targets.append(tokens[i, :length])

    targets = torch.cat(targets).to(device)
    print(f"Flattened targets shape: {targets.shape}")

    # Initialize model
    model = LipReadingModel(
        num_classes=55,
        hidden_size=256,
        dropout_rate=0.3,
        freeze_cnn=True
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # CTC Loss
    ctc_loss = nn.CTCLoss(
        blank=0,  # CTC blank token
        zero_infinity=True
    )

    model.train()

    # Forward pass
    outputs = model(frames)  # (T_max, B, num_classes)
    print(f"Model output shape: {outputs.shape}")

    # Log softmax for CTC
    log_probs = torch.log_softmax(outputs, dim=2)

    # CTC requires:
    # - log_probs: (T, B, C)
    # - targets: (sum of target lengths,)
    # - input_lengths: (B,) - actual frame counts
    # - target_lengths: (B,) - actual token counts

    loss = ctc_loss(
        log_probs,
        targets,
        frame_lengths,  # Actual frame counts per sample
        target_lengths  # Actual token counts per sample
    )

    print(f"Loss: {loss.item():.4f}")

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    print(" One training step completed successfully!")
    
    print(f"Loss is finite: {torch.isfinite(loss).item()}")


if __name__ == "__main__":
    main()

