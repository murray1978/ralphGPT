# Compressive Transformer, from the paper COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING,
# adapted from gptnv.py from comp_gpt.py from gpt_QA.py
# using this as a learing and testbed GPT.
# elapsed time incorrect, it's more like time accumulation.
# 6/10/24 tidy up and add json for models/parameters
#
# Idea's
#   Save the tokens with the model, include an unknown token
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

import matplotlib.pyplot as plt

import json

from tokeniser import Tokenizer

tokenizer = Tokenizer()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.memory.set_per_process_memory_fraction(0.9)

# hyperparameters
batch_size = 48 # 64 how many independent sequences will we process in parallel? '48 works'
block_size = batch_size * 4  # 256 what is the maximum context length for predictions?
max_iters = 8000
eval_interval = 400
min_val_loss = .90  # if validation loss below this value quit and save early, anything above 1.5 not good for inital training.
loss_separation = 0.3  # difference between val loss and train loss

# variable learning rate
learning_rate_fine = 0.9e-5 # 1e-5 
learning_rate = 2e-4  # 3e-4

# Transformer parameters
eval_iters = 100  # does not effect model
n_embd = 256  # effects model '256 works'
n_head = 10 # 6 effects model '10 works'
n_layer = 10  # 6 effects model '10 works'
dropout = 0.25  # does not effect model
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
model_filename = "gptnv2"
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
        print("Training mode selected.")
    elif choice == '2':
        _fine_tune = True
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

# Change to the tokenizer
# Additional Characters not found in the text
#additional_chars = r"<>[]{}123456780\?-+=#$" #

# here are all the unique characters that occur in this text
#chars = sorted(list(set(text) | set(additional_chars)))
#vocab_size = len(chars)

# create a mapping from characters to integers
# stoi = {ch: i for i,ch in enumerate(chars)}
# itos = {i: ch for i,ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print("adding special tokens to tokenizer")
tokenizer.append_special("__eou__")
tokenizer.append_special("。")
tokenizer.append_special('~')
tokenizer.append_special('‘')
tokenizer.append_special('¥')
tokenizer.append_special('£')
tokenizer.append_special('′')
tokenizer.append_special('°')
tokenizer.append_special('–')
tokenizer.append_special('“')
tokenizer.append_special('”')
tokenizer.append_special('\x7f')
tokenizer.append_special('、')  # at this point we should be modifiying tokenaizer.

# create a reference to the encoder and decoder
encode = tokenizer.encode
decode = tokenizer.decode
vocab_size = tokenizer.get_vocab_size()

if not for_chat: # do not need to set up a dataset if in chat
    print("preping dataset")
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]


# data loading
def get_batch(split, _data):
    # generate a small batch of data of inputs x and targets y
    _data = train_data if split == 'train' else val_data
    ix = torch.randint(len(_data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
            nn.LeakyReLU(negative_slope=negative_slope),
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
        stop_token = stop_token[-1]   #we should move this out of here.
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
        stop_token = stop_token[-1]  # Move this outside if needed

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

    if with_memory:
        output_ids = _model.generate_compressed(input_ids, max_new_tokens=max_new_tokens)
    else:
        output_ids = _model.generate_nv(input_ids, max_new_tokens=max_new_tokens, stop_token=tokenizer.getToken('__eou__'))

    # Get the token ID for "__eou__"
    eou_token_id = tokenizer.getToken('__eou__')
    eou_token_id = eou_token_id[-1]
    
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


model = GPTLanguageModel(max_memory_size=large_memory_size)

if os.path.exists(model_file):
    print(f'Using {model_file} ')
    model.load_state_dict(torch.load(model_file))
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_fine)
    fine_tune = True
else:
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

scaler = GradScaler()

training_loss = []
validation_loss = []
iteration_times = []

start_time = time.time()


# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)
timer = TrainingTimer( max_iters, eval_iters)

for iter in range(max_iters):
    iter_start_time = time.time()

    if for_chat:
        break

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        print("=" * 25)
        losses = estimate_loss()

        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        training_loss.append(losses['train'])
        validation_loss.append(losses['val'])

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
        iteration_times.append(iteration_time)

        # Calculate and display the total elapsed time
        total_elapsed_time = time.time() - start_time
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed time = {total_elapsed_time:.2f}s")

        # Update and get remaining time estimate
        remaining_time = timer.update()
        print(f"Estimated time remaining: {timer.format_time(remaining_time)}")


    # sample a batch of data
    xb, yb = get_batch('train', data)

    optimizer.zero_grad(set_to_none=True)

    # evaluate the loss
    with autocast():
        # xb = xb.requires_grad_(True)
        # yb = yb.requires_grad_(True)
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.empty_cache()
    gc.collect()


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


# Not happy with original chat, chatGPT 4o helped with the new one.
def chat_loop(model):
    user_input = ""
    print("Enter your queries. Type 'exit' to quit.")

    while True:
        try:
            _input = input(": ")
            if _input.lower() == "exit":
                print("Exiting chat.")
                break

            if _input.strip() == "":
                print("Input cannot be empty. Please try again.")
                continue

            # Combine the input and previous conversation context
            user_input += _input + " __eou__"

            # Generate response from the model
            response = generate_response(model, user_input, max_new_tokens=256)
            print(f"Model: {response}")

            # Update the context
            user_input += response + "\n"

        except TimeoutError:
            print("No input detected for a while. Auto-response generated.")
            response = generate_response(model, "No input detected.")
            print(f"Model: {response}")
            continue

def chat_loop2(model):
    user_input = ""
    print("Enter your queries. Type 'exit' to quit.")
    
    while True:
        _input = input(": ")
        if _input.lower() == "exit":
            print("Exiting chat.")
            break

        # Append new user input to conversation history
        user_input += f"User: {_input} __eou__"

        # Trim the conversation history to avoid excessive context
        user_input = trim_conversation_history(user_input)

        # Generate response from model
        response = generate_response(model, user_input, max_new_tokens=256)

        # Print model response
        print(f"Model: {response}")

        # Append model response to conversation history
        user_input += f"Model: {response} __eou__"

def chat_loop3(model):
    conversation_history = ""  # Keep a concise conversation history
    user_input_end = 0
    print("Enter your queries. Type '__exit__' to quit.")
    
    while True:
        _input = input("user: ")
        if _input.lower() == "__exit__":
            print("Exiting chat.")
            break

        # Append new user input to conversation history
        conversation_history += f"User: {_input} __eou__"

        user_input_length = len(f"User: {_input}")

        # Optionally limit the length of the conversation history if needed
        conversation_history = conversation_history[-1000:]  # Adjust this limit based on model capacity

        # Generate response from model
        response = generate_response(model, conversation_history, max_new_tokens=256)

        # Print model response
        #print(f"raw {response}")
        print(f"Model: {response[user_input_length:]}")

        # Append model response to conversation history
        conversation_history += f"Model: {response} __eou__"


if for_chat:
    chat_loop3(model=model)
