import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer, cer

from datasets.grid_dataset import GridDataset
from models.lipreading_model import LipReadingModel
from utils.tokenizer import Tokenizer


def flatten_targets(tokens, target_lengths):
    return torch.cat([tokens[i, :target_lengths[i].item()] for i in range(tokens.shape[0])])


def ctc_greedy_decode(log_probs, blank_idx=0):
    preds = torch.argmax(log_probs, dim=2).permute(1, 0)
    decoded = []

    for seq in preds:
        out, prev = [], None
        for idx in seq.tolist():
            if idx != blank_idx and idx != prev:
                out.append(idx)
            prev = idx
        decoded.append(out)

    return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

checkpoint_path = "/content/drive/MyDrive/LipStream/checkpoints_backup/lipreading_epoch_037.pth"

tokenizer = Tokenizer("data/vocab.txt")

test_dataset = GridDataset(
    "data/processed/lipread/train",
    "data/vocab.txt",
    img_size=112,
    split_file="data/splits/test.txt",
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=GridDataset.collate_fn,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

model = LipReadingModel(
    num_classes=55,
    hidden_size=256,
    dropout_rate=0.3,
    freeze_cnn=False
).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

total_loss = 0
total_wer = 0
total_cer = 0
word_correct = 0
word_total = 0
exact_correct = 0
count = 0

with torch.no_grad():
    for batch in test_loader:
        frames = batch["frames"].to(device)
        tokens_gt = batch["tokens"].to(device)
        frame_lengths = batch["frame_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        transcripts = batch["transcripts"]

        targets = flatten_targets(tokens_gt, target_lengths).to(device)

        outputs = model(frames)
        log_probs = torch.log_softmax(outputs, dim=2)

        loss = ctc_loss(log_probs, targets, frame_lengths, target_lengths)
        total_loss += loss.item()

        decoded = ctc_greedy_decode(log_probs)

        for i, tokens in enumerate(decoded):
            pred = tokenizer.decode(tokens)
            gt = transcripts[i]

            total_wer += wer(gt, pred)
            total_cer += cer(gt, pred)

            if pred == gt:
                exact_correct += 1

            pred_words = pred.split()
            gt_words = gt.split()

            for p, g in zip(pred_words, gt_words):
                if p == g:
                    word_correct += 1

            word_total += len(gt_words)
            count += 1

test_loss = total_loss / len(test_loader)
avg_wer = total_wer / count
avg_cer = total_cer / count
word_acc = word_correct / word_total
exact_acc = exact_correct / count

print("\n===== TEST RESULTS: EPOCH 37 =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Exact Sentence Accuracy: {exact_correct}/{count} = {exact_acc:.2%}")
print(f"Word Accuracy: {word_correct}/{word_total} = {word_acc:.2%}")
print(f"Average WER: {avg_wer:.4f}")
print(f"Average CER: {avg_cer:.4f}")
