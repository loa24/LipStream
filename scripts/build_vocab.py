#!/usr/bin/env python3

import os
from pathlib import Path

DATA_DIR = Path("data/processed/lipread/train")

vocab = set()
sample_count = 0
error_count = 0

print("🔍 Building vocabulary from transcripts...")
print(f"📁 Looking in: {DATA_DIR}\n")

# Iterate through all samples
for sample in sorted(os.listdir(DATA_DIR)):
    transcript_path = DATA_DIR / sample / "transcript.txt"
    
    # Check if file exists before opening
    if transcript_path.exists():
        try:
            with open(transcript_path, encoding='utf-8') as f:
                text = f.read().strip()
                
                # Convert to lowercase and split
                words = text.lower().split()
                
                # Add words to vocabulary
                vocab.update(words)
                
                sample_count += 1
                
                # Progress indicator
                if sample_count % 100 == 0:
                    print(f"  Processed {sample_count} samples...")
                    
        except Exception as e:
            print(f"  ️  Error reading {sample}: {e}")
            error_count += 1
    else:
        print(f"   Missing transcript: {transcript_path}")
        error_count += 1

# Sort vocabulary
vocab = sorted(list(vocab))

# Save vocabulary
output_file = Path("data/vocab.txt")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    for word in vocab:
        f.write(word + "\n")

# Print results
print(f"\n✓ Complete!")
print(f"  Samples processed: {sample_count}")
print(f"  Errors: {error_count}")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Saved to: {output_file}")
print(f"\n Sample words: {vocab[:10]}")

