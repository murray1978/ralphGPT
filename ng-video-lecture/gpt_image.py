# copy of GPT.py, but generating an image from the data input.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import gc
import os

# hyperparameters
batch_size = 64  # 64 how many independent sequences will we process in parallel?
block_size = batch_size * 4  # 256 what is the maximum context length for predictions?
max_iters = 2500  # 5000
eval_interval = 100
min_val_loss = 0.98  # if validation loss below this value quit and save early

# variable learning rate
learning_rate_fine = 1e-4
learning_rate = 3e-4  # 3e-4

# user conversation will have bits from, may not need brackets
gameLogicLabel = "[GL]"
playerLabel = "[PLAYER]"
startConvoLabel = "[START CONVERSATION]"
endConvoLabel = "[END CONVERSATION]"

# how to generate conversation context

# Data files
datafile = "input.txt"
modelfile = "game.pth"

# Training, tuning or chatting
fine_tune = False
for_chat = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

# Transformer parameters
eval_iters = 200
n_embd = 384
n_head = 6  # 6
n_layer = 6  # 6
dropout = 0.25
# ------------

torch.manual_seed(1337)


# ------------------------------------------------------------------
# Example function to generate pixel data using the model


def generate_pixel_data(model, context, encode, decode, max_pixels):
    context_tokens = encode(context)
    context_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = model.generate(context_tensor, max_new_tokens=max_pixels)
    generated_text = decode(generated_tokens[0].tolist())
    print(generated_text)

    # Extract pixel values from the generated text, preserving whitespace and newlines
    pixel_values = []
    for char in generated_text:
        if char in ' \n':
            pixel_values.append(0)  # Using 0 for spaces and newlines, adjust as needed
        else:
            pixel_values.append(ord(char))
   # print(pixel_values)

    # Ensure we have enough pixel values
    if len(pixel_values) < max_pixels:
        # Pad with zeros if not enough pixels
        pixel_values += [0] * (max_pixels - len(pixel_values))
    elif len(pixel_values) > max_pixels:
        # Trim to the required number of pixels
        pixel_values = pixel_values[:max_pixels]

    # Ensure the pixel values are in the range [0, 255]
    pixel_values = [min(max(0, value), 255) for value in pixel_values]
    # print(pixel_values)
    # Calculate image dimensions
    num_channels = 3  # Assuming RGB image
    height = width = int(np.sqrt(len(pixel_values) // num_channels))

    return np.array(pixel_values).reshape((height, width, num_channels))
# Example usage
# context = "[INPUT] Generate image [OUTPUT]"
# pixel_data = generate_pixel_data(model, context, encode, decode, block_size, 320 * 320 * 3)

def create_ppm_image(filename, pixel_data):
    height, width, _ = pixel_data.shape
    max_val = 255  # Max value for a pixel (PPM format uses values from 0 to 255)

    with open(filename, 'w') as f:
        # PPM Header
        f.write(f"P3\n{width} {height}\n{max_val}\n")

        # Pixel data
        for row in pixel_data:
            for pixel in row:
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
            f.write("\n")


# Create the PPM image file
# ppm_filename = "output_image.ppm"
# create_ppm_image(ppm_filename, pixel_data)
# print(f"PPM image created: {ppm_filename}")

# ------------------------------------------------------------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(datafile, 'r', encoding='utf-8') as f:
    text = f.read()

# Additional Characters
additional_chars = '[]{}'

# here are all the unique characters that occur in this text
chars = sorted(list(set(text) | set(additional_chars)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
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
            X, Y = get_batch(split)
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

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
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

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# Define a function to generate responses
def generate_response(model, query, max_new_tokens=100):
    model.eval()
    input_ids = torch.tensor(encode(query), dtype=torch.long).unsqueeze(0).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    response = decode(output_ids[0].tolist())
    return response


model = GPTLanguageModel()
if os.path.exists(modelfile):
    print(f'Using {modelfile} ')
    model.load_state_dict(torch.load(modelfile))
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_fine)
    max_iters = 1000
    fine_tune = True
else:
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

scaler = GradScaler()

for iter in range(max_iters):
    if for_chat:
        break

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < min_val_loss:
            print(f"Val loss {losses['val']} below {min_val_loss}, exit train loop")
            break

    # sample a batch of data
    xb, yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)

    # evaluate the loss
    with autocast():
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.empty_cache()
    gc.collect()
    #  loss.backward()
    #  optimizer.step()

# Save the model
torch.save(model.state_dict(), modelfile)

if fine_tune and not for_chat:
    # Example queries
    query = "Q: Who is Romeo?\nA:"
    response = generate_response(model, query)
    print(response)
elif for_chat:
    n = 0
    while True:
        user_input = input(":")
        pixel_data = generate_pixel_data(model, user_input, encode, decode,  320 * 320 * 3)
        ppm_filename = f"output_image{n}.ppm"
        create_ppm_image(ppm_filename, pixel_data)
        print(f"PPM image created: {ppm_filename}")
        # print(generate_response(model, user_input))
        n += 1
else:
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

