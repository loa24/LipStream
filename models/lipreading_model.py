import torch
import torch.nn as nn
import torchvision.models as models


class LipReadingModel(nn.Module):
    """
    Lip-reading model combining CNN + Bidirectional LSTM + CTC.
    
    Architecture:
    - ResNet-18 (visual feature extraction)
    - Bidirectional LSTM (temporal modeling)
    - Linear classifier (token prediction)
    - CTC Loss compatible output format
    """
    
    def __init__(
        self, 
        num_classes, 
        hidden_size=256,
        dropout_rate=0.3,
        freeze_cnn=True
    ):
        super().__init__()
        
        # Load ResNet-18 backbone with pretrained ImageNet weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_dim = 512
        
        # Remove classification head 
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze CNN weights
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval()
        
        # Dropout after CNN features
        self.dropout_cnn = nn.Dropout(dropout_rate)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout after LSTM
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        # Token classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
    
    def forward(self, frames):
        """
        Forward pass.
        
        Args:
            frames: (B, T, 3, 112, 112) - mouth crop sequences
        
        Returns:
            logits: (T, B, num_classes) - CTC compatible format
        """
        B, T, C, H, W = frames.shape
        
        # Reshape for CNN: process all frames as batch
        x = frames.view(B * T, C, H, W)
        
        # CNN feature extraction
        x = self.cnn(x)
        x = x.flatten(1)  # Explicitly flatten spatial dims
        
        # Reshape back to sequence format
        x = x.view(B, T, self.feature_dim)
        
        # Apply dropout to features
        x = self.dropout_cnn(x)
        
        # Bidirectional LSTM
        x, (h_n, c_n) = self.lstm(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Dropout after LSTM
        x = self.dropout_lstm(x)
        
        # Token classifier
        x = self.fc(x)
        
        # Permute for CTC loss: (T, B, C)
        x = x.permute(1, 0, 2)
        
        return x
    
    def get_feature_dimension(self):
        """Return CNN feature dimension."""
        return self.feature_dim


# Usage example:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LipReadingModel(
        num_classes=55,
        hidden_size=256,
        dropout_rate=0.3,
        freeze_cnn=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_length = 4, 75
    dummy_input = torch.randn(batch_size, seq_length, 3, 112, 112).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (seq_length, batch_size, 55), "Output shape mismatch!"
    print(" Model test passed!")

