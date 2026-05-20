from pathlib import Path

class Tokenizer:
    BLANK = "<blank>"
    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self, vocab_file, add_special_tokens=True, verbose=False):
        vocab_file = Path(vocab_file)

        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

        if add_special_tokens:
            words = [self.BLANK, self.UNK, self.PAD] + words

        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.blank_idx = self.word2idx[self.BLANK]
        self.unk_idx = self.word2idx[self.UNK]
        self.pad_idx = self.word2idx[self.PAD]

        if verbose:
            print(f"✓ Tokenizer loaded: {self.vocab_size()} tokens")

    def encode(self, text, handle_unknown="unk"):
        """
        Convert text to token indices
        
        Args:
            text: Input text (e.g., "hello world")
            handle_unknown: How to handle unknown words
                - "unk": Replace with <unk> token
                - "skip": Skip unknown words
                - "raise": Raise error (strict mode)
        
        Returns:
            List of token indices
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")

        words = text.lower().strip().split()
        indices = []
        unknown_count = 0

        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
               
                if handle_unknown == "unk":
                    indices.append(self.unk_idx)
                    unknown_count += 1
                elif handle_unknown == "skip":
                    unknown_count += 1
                    continue
                elif handle_unknown == "raise":
                    raise ValueError(f"Unknown word: '{word}'")
                else:
                    raise ValueError(f"Invalid handle_unknown: {handle_unknown}")

        if unknown_count > 0 and handle_unknown == "raise":
   	     print(f"  Found {unknown_count} unknown words in: '{text}'")

        return indices 
    def decode(self, indices, remove_special=True):
        """Convert token indices back to text"""
        if hasattr(indices, "tolist"):              indices = indices.tolist()

        words = []
        for idx in indices:
            if idx not in self.idx2word:
                continue

            word = self.idx2word[idx]

            # Skip special tokens if requested
            if remove_special and word in {self.BLANK, self.UNK, self.PAD}:
                continue

            words.append(word)

        return " ".join(words)

    def vocab_size(self):
        """Returns vocabulary size (including special tokens)"""
        return len(self.word2idx)

    def get_blank_idx(self):
        """Returns blank token index (for CTC loss)"""
        return self.blank_idx

    def get_num_classes(self):
        """Returns number of classes (for CTC loss configuration)"""
        return self.vocab_size()
