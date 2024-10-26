# Compressive Transformer, from the paper COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING,
# adapted from gptnv01.py from comp_gpt.py from gpt_QA.py
# using this as a learing and testbed GPT.
# elapsed time incorrect, it's more like time accumulation.
#
# 24/10/24 Change of tokenizer from char based to word/subword based
# 26/10/24 added frezze layers to code
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs except errors

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import gc
import os
import signal

from TrainingTimer import TrainingTimer
import time

import json

# from tokeniser import Tokenizer
from tokenizers import Tokenizer

# tokenizer = Tokenizer()
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.memory.set_per_process_memory_fraction(0.9)

# hyperparameters
batch_size = 32 # 64 how many independent sequences will we process in parallel? '48 works'
block_size = batch_size * 4  # 256 what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
min_val_loss = 1.4  # if validation loss below this value quit and save early, anything above 1.5 not good for inital training.
loss_separation = 5.3  # difference between val loss and train loss

# variable learning rate
learning_rate_fine = 1e-5 # 1e-5 
learning_rate = 3e-4  # 3e-4

# Transformer parameters, effects as in size, saving.
eval_iters = 100  # does not effect model
n_embd = 256 * 2   # effects model '256 works'
n_head = 10 # 6 effects model '10 works'
n_layer = 10  # 6 effects model '10 works'
dropout = 0.25  # does not effect model 0.25 'original'
# ------------

with_memory = False

# user conversation labels, to be added at a future date
labels = {
    "userLabel": "</user text=\"\">",
    "adminLabel": "</admin text=\"\">",
    "content_start":  "<content>",
    "content_end": "</content>",
    "convo_start": "<conversation>",
    "convo_end": "</conversation>",
    "character": "<character name=\"\">",
    "statementStart": "<statement>",
    "statementEnd": "</statement>",
}

model_params = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
}

# Data files
data_folder = "datasets/"
datafile = "datasets/dataset/ijcnlp_dailydialog/dialogues_text.txt"
# datafile = data_folder + "input-formatedFull.txt"
# model files
model_path = "models/"
model_ext = ".pth"
model_filename = "ralphGPT"
model_file = model_path + model_filename + model_ext
save_file = model_path + model_filename + "" + model_ext
# parameter file
param_file = model_path + model_filename + ".json"

#lets load the parameters
if os.path.exists(param_file):
    print(f"json file {param_file} found, loading")
else:
    print(f"json file {param_file} not found, creating")

# comment out for now may use it with second GPU
# preprocessor_model = model_folder + "preprocessor_model.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')


def signal_handler(sig, frame):
    if not for_chat:
        print('Ctrl-C detected, Saving model')
        if model is not None:
            torch.save(model.state_dict(), save_file)
            print(f'Model saved as {save_file}')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def select_mode():
    print("Select the mode you want to run:")
    print("1. Train")
    print("2. Fine-Tune")
    print("3. Chat")

    choice = input("Enter the number corresponding to your choice: ")

    _fine_tune = False
    _for_chat = False
    _train_only = False

    if choice == '1':
        _train_only = True
        dropout = 0.25
        print("Training mode selected.")
    elif choice == '2':
        _fine_tune = True
        dropout = 0.15      # Ok probably not the best place for it here
        print("Fine-Tuning mode selected.")
    elif choice == '3':
        _for_chat = True
        print("Chat mode selected.")
    else:
        print("Invalid choice. Defaulting to Training mode.")
        _train_only = True

    return _fine_tune, _for_chat, _train_only


# Training, tuning or chatting, we run out of memory if we train then chat.
# Select the mode
fine_tune, for_chat, train_only = select_mode()

small_memory_size = 512
medium_memory_size = 1024
large_memory_size = 2048 + 1024

torch.manual_seed(1337)

# Open up the inputfile for the tokenizer
with open(datafile, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('\n','')     

# decode for tokenizer
def decode(text):
    return tokenizer.decode(text)

# encode for tokenizer
def encode(text_ids):
    return tokenizer.encode(text_ids).ids

vocab_size = tokenizer.get_vocab_size()

# prints our check note "tokenizer" == "token izer" extra space for some reaseon
print( decode(encode("Check of tokenizer __eou__")))

def split_data(text, eou_token_id):
    if not for_chat:  # Do not need to set up a dataset if in chat
        print("Preparing dataset")

        # Tokenize the text
        data = torch.tensor(encode(text), dtype=torch.long)

        # Base dataset handling (no special treatment)
        if not fine_tune:
            print("\tBase dataset")
            n = int(0.9 * len(data))  # First 90% will be train, rest val
            train_data = data[:n]
            val_data = data[n:]
            return train_data, val_data

        # Fine-tune dataset handling (aligning with __eou__ tokens)
        else:
            print("\tFine-tune dataset")

            # Ensure eou_token_id is a scalar (not a list)
            if isinstance(eou_token_id, list):
                eou_token_id = eou_token_id[0]

            # Find indices where the __eou__ token appears
            eou_indices = (data == eou_token_id).nonzero(as_tuple=True)[0]

            # Split at the last complete conversation (aligned with __eou__)
            split_idx = int(0.9 * len(eou_indices))  # First 90% of conversations
            last_eou_idx = eou_indices[split_idx] + 1

            # Ensure that the split is even and aligned with sentence boundaries
            if last_eou_idx % 2 != 0:
                last_eou_idx -= 1  # Adjust to keep the split even

            # Create train and validation sets
            train_data = data[:last_eou_idx + 1]
            val_data = data[last_eou_idx + 1:]

            # Decode and print the train and validation sets
            """
            print("train_data:", decode(train_data.tolist()))
            print("="*50)
            print("val_data:", decode(val_data.tolist()))
            print("="*50)
            """
            return train_data, val_data

        
if not for_chat:
    train_data, val_data = split_data(text=text, eou_token_id=encode('__eou__'))

def print_tensor_size(tensor, tensor_name):
    num_elements = tensor.numel()  # Total number of elements in the tensor
    dtype_size = tensor.element_size()  # Size of each element in bytes (e.g., 4 bytes for float32)
    memory_usage = num_elements * dtype_size  # Total memory usage in bytes
    
    # Convert to more readable format (KB, MB, GB)
    memory_usage_kb = memory_usage / 1024
    memory_usage_mb = memory_usage_kb / 1024
    
    print(f"Memory usage of {tensor_name}: {memory_usage_mb:.2f} MB ({memory_usage_kb:.2f} KB)")


# data loading, relies on train_data and val_data setup outside of function
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    _data = train_data if split == 'train' else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([_data[i:i+block_size] for i in ix])
    y = torch.stack([_data[i+1:i+block_size+1] for i in ix])

    # Print memory usage before moving to GPU
    # print_tensor_size(x, 'x (input)')
    # print_tensor_size(y, 'y (output)')

    x, y = x.to(device), y.to(device)
    return x, y

def pad_or_trim(sequence, length):
    if len(sequence) > length:
        return sequence[:length]  # Trim if too long
    elif len(sequence) < length:
        # Pad with zeros if too short
        padding = torch.zeros(length - len(sequence), dtype=torch.long, device=sequence.device)
        return torch.cat([sequence, padding])
    return sequence

def get_train_pair_with_memory_limit(split, eou_token_id, memory_limit_kb=8):
    """
    Generates a small batch of input-output pairs for training, 
    ensuring that the memory usage of the batch does not exceed the given limit,
    and shuffles the data for each iteration. Ensures unique pairs.
    """
    # Calculate memory usage per token (long type = 8 bytes)
    token_size_bytes = torch.tensor(0, dtype=torch.long).element_size()  # Should be 8 bytes for long type

    # Use the global variables train_data or val_data depending on the split
    _data = train_data if split == 'train' else val_data
    
    if isinstance(eou_token_id, list):
        eou_token_id = eou_token_id[0]

    # Find the indices where the __eou__ token appears (marks end of a sentence)
    eou_indices = (_data == eou_token_id).nonzero(as_tuple=True)[0]

    # Shuffle the eou_indices for each iteration to get new pairs
    random.shuffle(eou_indices)

    # Initialize lists for input and target (user and bot) sentences
    input_sentences = []
    target_sentences = []
    seen_indices = set()  # Track seen indices to avoid duplicates
    prev_idx = 0
    batch_memory_usage = 0

    # Dynamic block size, starting at a high value and adjusting based on memory
    block_size = 64
    
    # Iterate over the shuffled indices and create user-bot pairs
    for idx in range(0, len(eou_indices) - 1, 2):
        if idx in seen_indices:  # Skip if pair already seen
            continue

        user_sentence = _data[prev_idx:eou_indices[idx] + 1]  # User input
        bot_sentence = _data[eou_indices[idx] + 1:eou_indices[idx + 1] + 1]  # Bot response

        # Check if the sentences are empty, and skip if they are
        if user_sentence.numel() == 0 or bot_sentence.numel() == 0:
            prev_idx = eou_indices[idx + 1] + 1
            continue  # Skip to the next pair if either sentence is empty

        # Add missing __eou__ token if not present
        if user_sentence[-1].item() != eou_token_id:
            user_sentence = torch.cat([user_sentence, torch.tensor([eou_token_id], dtype=torch.long)])
        if bot_sentence[-1].item() != eou_token_id:
            bot_sentence = torch.cat([bot_sentence, torch.tensor([eou_token_id], dtype=torch.long)])

        # Ensure both sentences are padded/truncated to block_size
        user_sentence = pad_or_trim(user_sentence, block_size)
        bot_sentence = pad_or_trim(bot_sentence, block_size)

        # Calculate the memory usage of the current batch
        current_memory_usage = (user_sentence.numel() + bot_sentence.numel()) * token_size_bytes
        
        # Check if adding this pair would exceed the memory limit
        if batch_memory_usage + current_memory_usage > memory_limit_kb * 1024:
            # Stop adding more pairs if memory limit is exceeded
            break

        input_sentences.append(user_sentence)
        target_sentences.append(bot_sentence)
        
        # Update batch memory usage
        batch_memory_usage += current_memory_usage

        # Mark these indices as seen
        seen_indices.add(idx)
        seen_indices.add(idx + 1)

        prev_idx = eou_indices[idx + 1] + 1  # Move to next pair
    
    # Stack the sentences into tensors for batch processing
    if len(input_sentences) > 0:
        x = torch.stack(input_sentences)  # Inputs (user sentences)
        y = torch.stack(target_sentences)  # Targets (bot sentences)
    
        # Print the final memory usage
        # print(f"Total batch memory usage: {batch_memory_usage / 1024:.2f} KB")
    
        # Move tensors to the device
        x, y = x.to(device), y.to(device)
    
        return x, y
    else:
        print("No valid data pairs were found for this iteration.")
        return None, None  # Return None if no data pairs are found

"""
    Data loading for fine tuning, this part of the training program is causing issues due to
    train/validate pair's being duplicated.
"""
def get_train_pair_with_memory_limit_old(split, eou_token_id, memory_limit_kb=16):
    """
    Generates a small batch of input-output pairs for training, 
    ensuring that the memory usage of the batch does not exceed the given limit.
    """
    # Calculate memory usage per token (long type = 8 bytes)
    token_size_bytes = torch.tensor(0, dtype=torch.long).element_size()  # Should be 8 bytes for long type

    # Use the global variables train_data or val_data depending on the split
    _data = train_data if split == 'train' else val_data
    
    if isinstance(eou_token_id, list):
        eou_token_id = eou_token_id[0]
    
    # Find the indices where the __eou__ token appears (marks end of a sentence)
    eou_indices = (_data == eou_token_id).nonzero(as_tuple=True)[0]
    
    # Initialize lists for input and target (user and bot) sentences
    input_sentences = []
    target_sentences = []
    prev_idx = 0
    batch_memory_usage = 0

    # Dynamic block size, starting at a high value and adjusting based on memory
    block_size = 64
    
    # Iterate over the indices and create user-bot pairs
    for idx in range(0, len(eou_indices) - 1, 2):
        user_sentence = _data[prev_idx:eou_indices[idx] + 1]  # User input
        bot_sentence = _data[eou_indices[idx] + 1:eou_indices[idx + 1] + 1]  # Bot response
        
        # Ensure both sentences are padded/truncated to block_size
        user_sentence = pad_or_trim(user_sentence, block_size)
        bot_sentence = pad_or_trim(bot_sentence, block_size)

        # Calculate the memory usage of the current batch
        current_memory_usage = (user_sentence.numel() + bot_sentence.numel()) * token_size_bytes
        
        # Check if adding this pair would exceed the memory limit
        if batch_memory_usage + current_memory_usage > memory_limit_kb * 1024:
            # Stop adding more pairs if memory limit is exceeded
            break

        input_sentences.append(user_sentence)
        target_sentences.append(bot_sentence)
        
        # Update batch memory usage
        batch_memory_usage += current_memory_usage

        prev_idx = eou_indices[idx + 1] + 1  # Move to next pair
    
    # Stack the sentences into tensors for batch processing
    x = torch.stack(input_sentences)  # Inputs (user sentences)
    y = torch.stack(target_sentences)  # Targets (bot sentences)
    
    # Print the final memory usage
    # print(f"Total batch memory usage: {batch_memory_usage / 1024:.2f} KB")
    
    # Move tensors to the device
    x, y = x.to(device), y.to(device)
    
    return x, y

"""
 This is chatGPT 4o's, third atempt, this is not the result of 
 chatGPT getting it wrong but a miss understanding in what the end
 user thought they specified and what the programmer thinks
 the end user has specified.
"""
def get_train_pair(split, eou_token_id):
    _block_size = 0
    """
    Generates a small batch of input-output pairs for training, 
    based on user-bot sentence pairs separated by __eou__ (end of utterance).
    """
    # Use the global variables train_data or val_data depending on the split
    _data = train_data if split == 'train' else val_data
    
    # Move data to the GPU early to avoid CPU overhead,
    # had ro 
    # _data = _data.to(device)
    
    # Ensure eou_token_id is a scalar (not a list)
    if isinstance(eou_token_id, list):
        eou_token_id = eou_token_id[0]
    
    # Find the indices where the __eou__ token appears (marks end of a sentence)
    eou_indices = (_data == eou_token_id).nonzero(as_tuple=True)[0]
    
    # Split the data into user and bot sentences based on __eou__ token positions
    input_sentences = []
    target_sentences = []
    prev_idx = 0
    
    # Iterate over the end-of-utterance indices and create user-bot pairs
    for idx in range(0, len(eou_indices) - 1, 2):  # Skip every other for user-bot pairs
        user_sentence = _data[prev_idx:eou_indices[idx] + 1]  # User input
        bot_sentence = _data[eou_indices[idx] + 1:eou_indices[idx + 1] + 1]  # Bot response
        
        # Ensure both sentences are padded/truncated to block_size
        user_sentence = pad_or_trim(user_sentence, _block_size)
        bot_sentence = pad_or_trim(bot_sentence, _block_size)
        
        input_sentences.append(user_sentence)
        target_sentences.append(bot_sentence)
        
        prev_idx = eou_indices[idx + 1] + 1  # Move to next pair
    
    # Stack the sentences into tensors for batch processing
    x = torch.stack(input_sentences)  # Inputs (user sentences)
    y = torch.stack(target_sentences)  # Targets (bot sentences)
    
    # Print memory usage before moving to GPU
    # print_tensor_size(x, 'x (input)')
    # print_tensor_size(y, 'y (output)')

    # Move tensors to the device
    x, y = x.to(device), y.to(device)
    
    return x, y

def freeze_layers(model, num_layers_to_frezze):

    # If model is wrapped in DataParallel, access the underlying model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # freeze embedding layer and first num_lauers_to_freeze transformer blocks
    for param in model.token_embedding_table.parameters():
        param.requires_grad = False # Freeze token embedding layer

    for param in model.position_embedding_table.parameters():
        param.requires_grad = False # Freeze position embedding layer
    
    # Freeze the first num_layers_to_freeze transformer blocks
    for i, block in enumerate(model.blocks):
        if i < num_layers_to_frezze:
            for param in block.parameters():
                param.requires_grad = False

    return model

def gradual_unfreeze(model, current_epoch, freeze_epoch_interval):
    # Ensure we access the underlying model if wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Get the total number of layers
    num_layers = len(model.blocks)
    
    # Calculate the number of layers to unfreeze based on the current epoch
    layers_to_unfreeze = min(current_epoch // freeze_epoch_interval, num_layers)
    
    # Unfreeze layers up to `layers_to_unfreeze`
    for i in range(layers_to_unfreeze):
        for param in model.blocks[i].parameters():
            param.requires_grad = True


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if train_only:
                X, Y = get_batch(split)
            else:
                X, Y = get_train_pair_with_memory_limit(split=split, eou_token_id=encode('__eou__'))
            logits, loss = model(X, Y)

            #handle multiple GPUS
            if torch.cuda.device_count() > 1:
                loss = loss.mean()

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # B
        self.query = nn.Linear(n_embd, head_size, bias=False) # T
        self.value = nn.Linear(n_embd, head_size, bias=False) # C
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, negative_slope=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # Leaky relu, helps prevent dead neurons.
            nn.LeakyReLU(negative_slope=negative_slope),
            #GELU, to slow for my liking
            # nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, compression_rate=2,  max_memory_size=1024):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.max_memory_size = max_memory_size
        # Compression rate and secondary memory
        if with_memory:
            self.compression_rate = compression_rate
            self.secondary_memory = None

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compress_activations(self, activations):
        _batch_size, seq_length, hidden_size = activations.shape

        # Calculate the padding length needed to make seq_length divisible by the compression_rate
        padding_length = (self.compression_rate - seq_length % self.compression_rate) % self.compression_rate

        # Pad the activations tensor with zeros (or any other padding strategy)
        if padding_length > 0:
            pad = torch.zeros(_batch_size, padding_length, hidden_size, device=activations.device)
            activations = torch.cat((activations, pad), dim=1)

        # Update the sequence length after padding
        seq_length = activations.shape[1]

        # Reshape the activations to (batch_size, seq_length // compression_rate, compression_rate, hidden_size)
        activations_reshaped = activations.view(_batch_size, seq_length // self.compression_rate, self.compression_rate,
                                                hidden_size)

        # Compute the mean of each group along the compression_rate dimension
        compressed_activations = activations_reshaped.mean(dim=2)

        return compressed_activations

    def forward(self, idx, targets=None):
        if with_memory:
            return self.forward_memory(idx, targets)
        else:
            return self.forward_normal(idx, targets)

    def forward_normal(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def forward_memory(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)

        # Compress the oldest activations if necessary
        if T > self.compression_rate:
            oldest_activations = x[:, :T - self.compression_rate, :]
            compressed_activations = self.compress_activations(oldest_activations)

            if self.secondary_memory is None:
                self.secondary_memory = compressed_activations
            else:
                # Ensure the secondary memory and compressed activations have compatible shapes
                if self.secondary_memory.shape[0] != compressed_activations.shape[0]:
                    raise ValueError("Batch size mismatch between secondary memory and compressed activations")
                if self.secondary_memory.shape[2] != compressed_activations.shape[2]:
                    raise ValueError("Hidden size mismatch between secondary memory and compressed activations")

                # Limit the size of the secondary memory to max_memory_size
                if self.secondary_memory.size(1) > self.max_memory_size:
                    self.secondary_memory = self.secondary_memory[:, -self.max_memory_size:, :]

                self.secondary_memory = torch.cat((self.secondary_memory, compressed_activations), dim=1)

            # print(f"x pre slice {x.shape}")
            x = x[:, T - self.compression_rate:, :]
            # print(f"x post slice {x.shape}")
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # print(f"logits shape: {logits.shape}")
            # print(f"targets shape: {targets.shape}")
            B, T, C = logits.shape
            targets = targets[:, -T:].contiguous()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        return logits, loss

    def generate(self, idx, max_new_tokens, stop_token=None):
        self.secondary_memory = None
        # stop_token = stop_token[-1]   #we should move this out of here.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            last_item = idx[:,-1].cpu().item()  # end token?
            if last_item == stop_token:     
                return idx
        return idx
  

    def generate_nv(self, idx, max_new_tokens, stop_token=None, temperature=1.0, top_p=0.9):
        self.secondary_memory = None
        # stop_token = stop_token[-1]  # Move this outside if needed

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            # Forward pass to get logits
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_probs = sorted_probs[cumulative_probs <= top_p]
                sorted_indices = sorted_indices[cumulative_probs <= top_p]
                            # Ensure there are tokens remaining to sample from
                if sorted_probs.size(-1) > 0:
                    # Normalize the remaining probabilities
                    probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)

                    # Sample from the filtered probabilities
                    idx_next = sorted_indices[torch.multinomial(probs, num_samples=1)]  # (B, 1)
                else:
                    # If no tokens are left after filtering, sample from the original distribution
                    idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            else:
                # Sample without top-p filtering
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Ensure idx_next is 2D
            if idx_next.dim() == 1:  # If idx_next is 1D, we need to unsqueeze
                idx_next = idx_next.unsqueeze(1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            # Check if the generated token matches the stop token
            last_item = idx[:, -1].cpu().item()
            if last_item == stop_token:
                return idx

        return idx


    def generate_compressed(self, idx, max_new_tokens):
        self.secondary_memory = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            # Update the secondary memory with new activations
            with torch.no_grad():
                # Recompute the activations for the entire sequence
                tok_emb = self.token_embedding_table(idx)  # (B,T,C)
                pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))  # (T,C)
                x = tok_emb + pos_emb  # (B,T,C)
                x = self.blocks(x)  # (B,T,C)
                compressed_activations = self.compress_activations(x[:, :idx.shape[1] - self.compression_rate, :])

                if self.secondary_memory is None:
                    self.secondary_memory = compressed_activations
                else:
                    self.secondary_memory = torch.cat((self.secondary_memory, compressed_activations), dim=1)

        return idx


# Define a function to generate responses
def generate_response(_model, _query, max_new_tokens=60):
    _model.eval()
    input_ids = torch.tensor(encode(_query), dtype=torch.long).unsqueeze(0).to(device)
    # Get the token ID for "__eou__"
    #eou_token_id = tokenizer.getToken('__eou__')
    eou_token_id = tokenizer.token_to_id('__eou__')
    #eou_token_id = eou_token_id[-1]

    if torch.cuda.device_count() > 1 :
        if with_memory:
            output_ids = _model.module.generate_compressed(input_ids, max_new_tokens=max_new_tokens)
        else:
            output_ids = _model.module.generate_nv(input_ids, max_new_tokens=max_new_tokens, stop_token=eou_token_id)
    else:
        if with_memory:
            output_ids = _model.generate_compressed(input_ids, max_new_tokens=max_new_tokens)
        else:
            output_ids = _model.generate_nv(input_ids, max_new_tokens=max_new_tokens, stop_token=eou_token_id)

   
    # Check if output_ids is a tensor
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.tolist()  # Convert tensor to a list
    
    # Remove any occurrences of the "__eou__" token from output_ids
    output_ids = [token for token in output_ids[0] if token != eou_token_id]  # Process the sequence
    
    # return decode(output_ids[0].tolist())
    return decode(output_ids)

import random
import string

def generate_random_chars(length=25):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

import threading

def input_with_timeout(prompt, timeout):
    user_input = []

    def get_input():
        user_input.append(input(prompt))

    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None  # If timeout reached, return None
    return user_input[0]


# Initialize the model and move it to device
model = GPTLanguageModel(max_memory_size=large_memory_size).to(device)

# Load model state if the file exists, without wrapping in DataParallel initially
if os.path.exists(model_file):
    print(f'Loading {model_file} ')
    state_dict = torch.load(model_file)

    # Remove 'module.' prefix if it exists in the state_dict keys
    if "module." in list(state_dict.keys())[0]:  # Check if prefix exists
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load the state dict into the model (not wrapped in DataParallel yet)
    model.load_state_dict(state_dict)
    print('\tDone loading model.')

# Wrap model in DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPU(s)")
    model = nn.DataParallel(model)

# Set up optimizer after DataParallel and loading
if fine_tune:
    # Freeze layers before creating the optimizer
    model = freeze_layers(model=model, num_layers_to_frezze=8)

    # Check which layers are frozen (i.e., requires_grad is False)
    frozen_layers = sum(not param.requires_grad for param in model.parameters())
    total_layers = sum(1 for _ in model.parameters())
    print(f"{frozen_layers} out of {total_layers} layers are frozen.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_fine)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Ensure model is on the correct device (optional as it's already to(device))
model = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

scaler = GradScaler()

start_time = time.time()


# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)
timer = TrainingTimer( max_iters, eval_iters)

# train the from the dataset normally
def train_normal():

    for iter in range(max_iters):
        iter_start_time = time.time()

        if for_chat:
            break

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print("=" * 25)
            print(f"Iteration {iter}, start time {timer.format_time(iter_start_time)}")
            losses = estimate_loss()

            if losses['val'] < min_val_loss:
                print(f"Val loss {losses['val']} below {min_val_loss}, exit train loop")
                break

            separation = losses['val'] - losses['train']
            if separation > loss_separation :
                print(f"Loss Separation {separation} below {loss_separation}, exit train loop")
                break

            scheduler.step(losses['val'])

            # Calculate elapsed time per iteration
            iter_end_time = time.time()
            iteration_time = iter_end_time - iter_start_time

            # Calculate and display the total elapsed time
            total_elapsed_time = time.time() - start_time
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed time = {total_elapsed_time:.2f}s")

            # Update and get remaining time estimate
            remaining_time = timer.update()
            print(f"Estimated time remaining: {timer.format_time(remaining_time)}")


        # sample a batch of data
        xb, yb = get_batch('train')

        optimizer.zero_grad(set_to_none=True)

        # evaluate the loss
        with autocast():
            # xb = xb.requires_grad_(True)
            # yb = yb.requires_grad_(True)
            logits, loss = model(xb, yb)

        if torch.cuda.device_count() > 1:
            if loss.dim() > 0:
                loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        gc.collect()

def train_fine_tune():
    for iter in range(max_iters):
        iter_start_time = time.time()

        if for_chat:
            break

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print("=" * 25)
            print(f"Iteration {iter}, start time {timer.format_time(iter_start_time)}")
            losses = estimate_loss()

            if losses['val'] < min_val_loss:
                print(f"Val loss {losses['val']} below {min_val_loss}, exit train loop")
                break

            separation = losses['val'] - losses['train']
            if separation > loss_separation :
                print(f"Loss Separation {separation} below {loss_separation}, exit train loop")
                break

            scheduler.step(losses['val'])

            # Calculate elapsed time per iteration
            iter_end_time = time.time()
            iteration_time = iter_end_time - iter_start_time

            # Calculate and display the total elapsed time
            total_elapsed_time = time.time() - start_time
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed time = {total_elapsed_time:.2f}s")

            # Update and get remaining time estimate
            remaining_time = timer.update()
            print(f"Estimated time remaining: {timer.format_time(remaining_time)}")


        # sample a batch of data
        # xb, yb = get_batch('train', data)
        eou_token_id = encode('__eou__')
        xb, yb = get_train_pair_with_memory_limit('train', eou_token_id)

        optimizer.zero_grad(set_to_none=True)

        # evaluate the loss
        with autocast():
            # xb = xb.requires_grad_(True)
            # yb = yb.requires_grad_(True)
            logits, loss = model(xb, yb)

        if torch.cuda.device_count() > 1:
            if loss.dim() > 0:
                loss = loss.mean()

        gradual_unfreeze(model, current_epoch=iter, freeze_epoch_interval=5)
            # Check which layers are frozen (i.e., requires_grad is False)
        frozen_layers = sum(not param.requires_grad for param in model.parameters())
        total_layers = sum(1 for _ in model.parameters())
        print(f"{frozen_layers} out of {total_layers} layers are frozen.")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        gc.collect()

if fine_tune:
    print("training fine tune")
    train_fine_tune()
else:
    print("Training base data")
    train_normal()

if not for_chat:
    print(f"Saving {save_file}")
    # Save the model
    torch.save(model.state_dict(), save_file)
    print(f"Saving json file {param_file} for model params")
    with open("test.json", 'w') as f:
        json.dump(model_params, f)

MAX_HISTORY_TOKENS = 512  # You can adjust this value based on the model's block size

def trim_conversation_history(conversation, max_length=MAX_HISTORY_TOKENS):
    """
    Trim the conversation history to ensure it doesn't exceed the max token length.
    """
    # Tokenize conversation to count tokens
    tokens = tokenizer.encode(conversation)
    if len(tokens) > max_length:
        # Truncate the oldest part of the conversation
        trimmed_tokens = tokens[-max_length:]
        return tokenizer.decode(trimmed_tokens)
    return conversation

## still not there, but getting closer.
def chat_loop5(model):
    conversation_history = []  # Keep recent exchanges only
    max_history_pairs = 1  # Keep only the last exchange for context
    total_len = 0
    print("Enter your queries. Type '__exit__' to quit.")
    
    while True:
        _input = input("user: ")
        if _input.lower() == "__exit__":
            print("Exiting chat.")
            break

        input_length = len(_input)
        _input += f"{_input} __eou__"
        
        print(f"input length {input_length}")

        # Append the new user input to conversation history
        conversation_history.append(_input)

        # Limit conversation history to avoid echo and repetition
        if len(conversation_history) > max_history_pairs * 2:
            conversation_history = conversation_history[-max_history_pairs * 2:]

        # Prepare input with limited context for the model
        conversation_str = " ".join(conversation_history)

        # Generate response
        response = generate_response(model, conversation_str, max_new_tokens=256)

        # Trim response to avoid echoing and extraneous parts
        model_response = response.strip()
        response_len = len(model_response)
        total_len = total_len + response_len + input_length

        print(f"respose length {response_len}")
        print(f"end position is {total_len - response_len}")

        # Display the current user input and model response only
        print(f"Model: {model_response[total_len - response_len:]}")

        # Add model response to conversation history for minimal context
        conversation_history.append(f" {model_response} __eou__")

def chat_loop6(model):
    conversation_history = []  # Keep recent exchanges only
    max_history_pairs = 1  # Keep only the last exchange for context
    total_len = 0
    print("Enter your queries. Type '__exit__' to quit.")
    
    while True:
        _input = input("user: ")
        if _input.lower() == "__exit__":
            print("Exiting chat.")
            break

        input_length = len(_input)
        _input += f"{_input} __eou__"
        
        print(f"input length {input_length}")

        # Append user input to conversation history
        conversation_history.append(_input)

        # Limit conversation history to the last `max_history_pairs` exchanges
        if len(conversation_history) > max_history_pairs * 2:
            conversation_history = conversation_history[-max_history_pairs * 2:]

        # Compile limited conversation history for model context
        conversation_str = " ".join(conversation_history)

        # Generate response from model
        response = generate_response(model, conversation_str, max_new_tokens=256).strip()
        

        # Trim any echo of user input at the start of the response

        model_response = response[len(_input):].strip()
        response_len = len(model_response)

        print(f"respose length {response_len}")
        print(f"end position is {total_len - response_len}")


        # Print the user input and trimmed model response
        print(f"Model raw: {model_response}")
        print(f"Model: {model_response[total_len - response_len:]}")

        # Append model response to conversation history
        conversation_history.append(f"{model_response} __eou__")

        total_len = total_len + response_len + input_length


if for_chat:
    chat_loop6(model=model)
