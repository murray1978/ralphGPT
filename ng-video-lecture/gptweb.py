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

#from tokeniser import Tokenizer
from tokenizers import Tokenizer
from flask import Flask, request, jsonify

#tokenizer = Tokenizer()
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
max_iters = 8000
eval_interval = 400
min_val_loss = .90  # if validation loss below this value quit and save early, anything above 1.5 not good for inital training.
loss_separation = 0.3  # difference between val loss and train loss

# variable learning rate
learning_rate_fine = 1e-5 # 1e-5 
learning_rate = 2e-4  # 3e-4

# Transformer parameters
eval_iters = 100  # does not effect model
n_embd = 256 * 2 # effects model '256 works'
n_head = 10 # 6 effects model '10 works'
n_layer = 10  # 6 effects model '10 works'
dropout = 0.25  # does not effect model
# ------------

with_memory = False

# how to generate conversation context
model_folder = "models/"
model_file = model_folder + "ralphGPTweb01.pth"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')


def signal_handler(sig, frame):

    print('Ctrl-C detected')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

torch.manual_seed(1337)

def decode(text):
    return tokenizer.decode(text)

def encode(text_ids):
    return tokenizer.encode(text_ids).ids
    
vocab_size = tokenizer.get_vocab_size()

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
        #stop_token = stop_token[-1]   #we should move this out of here.
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
    eou_token_id = tokenizer.token_to_id('__eou__')

    if with_memory:
        output_ids = _model.generate_compressed(input_ids, max_new_tokens=max_new_tokens)
    else:
        output_ids = _model.generate(input_ids, max_new_tokens=max_new_tokens, stop_token=eou_token_id)

    # Get the token ID for "__eou__"
    # eou_token_id = tokenizer.getToken('__eou__')
    # eou_token_id = eou_token_id[-1]
    
    # Check if output_ids is a tensor
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.tolist()  # Convert tensor to a list
    
    # Remove any occurrences of the "__eou__" token from output_ids
    output_ids = [token for token in output_ids[0] if token != eou_token_id]  # Process the sequence
    
    # return decode(output_ids[0].tolist())
    return decode(output_ids)


model = GPTLanguageModel()

if os.path.exists(model_file):
    print(f'Using {model_file} ')
    # model.load_state_dict(torch.load(model_file))
    # Load the saved state_dict
    state_dict = torch.load(model_file)

    # Remove 'module.' prefix from the keys, since we trained on multi GPU's
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove the 'module.' prefix
        new_state_dict[new_key] = v

    # Load the modified state_dict into your model
    model.load_state_dict(new_state_dict)

else:
    print(f"{model_file} not found")
    exit()

# Wrap model in DataParallel if multiple GPUs are available
#if torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPU(s)")
#    model = nn.DataParallel(model)

model = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

def generateChat(input):
    input += input + " __eou__"
    response = generate_response(model, input, max_new_tokens=256)
    input += response + "\n"
    return response

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <form action="/generate" method="post">
        <textarea name="prompt" rows="4" cols="50" placeholder="Message RalphGPT"></textarea><br>
        <input type="submit" value="Generate Response">
    </form>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    response = generateChat(prompt)  # Call your GPT model's generate function
    return f"<h2>Response:</h2><p>{response}</p><br><a href='/'>Go Back</a>"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
