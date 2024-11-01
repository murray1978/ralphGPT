from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# Step 1: Initialize a Byte-Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Step 2: Set up the trainer
trainer = BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "__eou__","__usr__","__bot__"])

# Step 3: Train the tokenizer on your dataset
datafile = "datasets/dataset/ijcnlp_dailydialog/dialogues_text.txt"
if os.path.exists(datafile):
    with open(datafile, "r", encoding="utf-8") as f:
        lines = f.readlines()

tokenizer.train_from_iterator(lines, trainer)

# Save the tokenizer to disk
tokenizer.save("ralphGPTv4_tokenizer.json")
