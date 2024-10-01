# Compressive Transformer, from the paper COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING,
# adapted from comp_gpt.py from gpt_QA.py
# using this as a learing and testbed GPT.
# elapsed time incorrect, it's more like time accumulation.
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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.memory.set_per_process_memory_fraction(0.9)

# hyperparameters
batch_size = 48 # 64 how many independent sequences will we process in parallel? '48 works'
block_size = batch_size * 4  # 256 what is the maximum context length for predictions?
max_iters = 600
eval_interval = 100
min_val_loss = 1.289  # if validation loss below this value quit and save early
loss_separation = 0.5  # difference between val loss and train loss

# variable learning rate
learning_rate_fine = 1e-5
learning_rate = 2e-4  # 3e-4

# Transformer parameters
eval_iters = 100  # does not effect model
n_embd = 256  # effects model '256 works'
n_head = 10 # 6 effects model '10 works'
n_layer = 10  # 6 effects model '10 works'
dropout = 0.25  # does not effect model
# ------------

with_memory = False

# user conversation will have bits from, may not need brackets
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
# how to generate conversation context

# Data files
data_folder = "dataset"
datafile = "input-formatedFull.txt"
model_folder = "models"
model_file = "gptnv_normal21a.pth"
save_file = "gptnv_normal21a.pth"
preprocessor_model = "preprocessor_model.pth"

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
    print("4. Show Loss")
    print("5. Show Graphs")

    choice = input("Enter the number corresponding to your choice: ")

    _fine_tune = False
    _for_chat = False
    _train_only = False
    _print_loss = False
    _show_graphs = False
    _interativePlot = False

    if choice == '1':
        _train_only = True
        _interativePlot = True
        print("Training mode selected.")
    elif choice == '2':
        _fine_tune = True
        _interativePlot = True
        print("Fine-Tuning mode selected.")
    elif choice == '3':
        _for_chat = True
        print("Chat mode selected.")
    elif choice == '4':
        _print_loss = True
        print("Show Loss mode selected.")
    elif choice == '5':
        _show_graphs = True
        print("Show Graphs mode selected.")
    else:
        print("Invalid choice. Defaulting to Training mode.")
        _train_only = True

    return _fine_tune, _for_chat, _train_only, _print_loss, _show_graphs, _interativePlot


# Training, tuning or chatting, we run out of memory if we train then chat.
# Select the mode
fine_tune, for_chat, train_only, print_loss, show_graphs, interativePlot = select_mode()

small_memory_size = 512
medium_memory_size = 1024
large_memory_size = 2048 + 1024

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(datafile, 'r', encoding='utf-8') as f:
    text = f.read()                          # change to the tokenizer.

# Additional Characters not found in the text
additional_chars = r"<>[]{}123456780\?-+=#$" #

# here are all the unique characters that occur in this text
chars = sorted(list(set(text) | set(additional_chars)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

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

    def generate(self, idx, max_new_tokens):
        self.secondary_memory = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
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
        output_ids = _model.generate(input_ids, max_new_tokens=max_new_tokens)
    return decode(output_ids[0].tolist())

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

if interativePlot:
    plt.ion()
    fig, ax = plt.subplots()

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

        if interativePlot:
            # Update the plot
            ax.clear()
            ax.plot(training_loss, label='Training Loss')
            ax.plot(validation_loss, label='Validation Loss')
            ax.set_xlabel('Evaluation Intervals')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss Over Time')
            ax.legend()
            plt.pause(0.01)  # Pause to update the plot

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

if interativePlot:
    # After training, keep the plot open
    plt.ioff()
    plt.show()

if not for_chat:
    print(f"Saving {save_file}")
    # Save the model
    torch.save(model.state_dict(), save_file)

if show_graphs:
    import matplotlib.pyplot as plt

    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Evaluation Intervals')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()

if fine_tune and not for_chat:
    # Example queries
    query = "[Q]: Who is Romeo?\n[A]:"
    response = generate_response(model, query)
    print(response)

elif for_chat:
    user_input = ""
    while True:
        _input = input(":")
        # _input = input_with_timeout(":", timeout=10)  # Timeout after 5 seconds
        if _input:
            user_input += "<character name=user><question>" + _input + "</question></character>"
        else:
            user_input += generate_random_chars() + "?"
        response = generate_response(model, user_input, max_new_tokens=256)
        print(response)
        user_input += response + "\n"


else:
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
