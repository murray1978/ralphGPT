import os
import signal
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
from tokenizer import Tokenizer
from gpt_model import Params
from gpt_model import GPTLanguageModel
from TrainingTimer import TrainingTimer
import tensorflow as tf
import threading
import time
import gc

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.memory.set_per_process_memory_fraction(0.9)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs except errors

model_parameters = {
    "batch_size": 48,  # 64 how many independent sequences will we process in parallel? '48 works'
    "block_size": 0,
    "max_iters": 2500,
    "eval_iterations": 100,
    "eval_interval": 100,
    "min_val_loss": 1.389,  # if validation loss below this value quit and save early
    "loss_separation": 0.6,  # difference between val loss and train loss
    # variable learning rate
    "learning_rate_fine": 1e-5,
    "learning_rate": 2e-4,  # 3e-4
    # Transformer parameters
    "eval_iters": 100,  # does not effect model
    "n_embd":  256,  # effects model '256 works'
    "n_head": 10, # 6 effects model '10 works'
    "n_layer": 10,  # 6 effects model '10 works'
    "dropout": 0.25,  # does not effect model
    # Memory parameters
    "small_memory_size": 512,
    "medium_memory_size": 1024,
    "large_memory_size": 2048 + 1024,
    # data and model parameters
    "with_memory": False,
    "data_dir": "data",
    "data_file": "input.txt",
    "model_dir": "model",
    "model_file": "shakespare.pth",
    "model_file_exists": False,
}

model_parameters['block_size'] = model_parameters['batch_size'] * 4
tokenizer = Tokenizer()

params = Params()
params.vocab_size = tokenizer.vocab_size


def signal_handler(sig, frame):
    #if not for_chat:
    #    print('Ctrl-C detected, Saving model')
    #    if model is not None:
    #        torch.save(model.state_dict(), modelfile)
    #        print(f'Model saved as {modelfile}')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def open_data_file( _datafile):
    with open(_datafile, 'r', encoding='utf-8') as f:
        _text = f.read()
    return _text


# data loading
def get_batch(_split, _dataset):
    _train_data, _val_data = _dataset
    # generate a small batch of data of inputs x and targets y
    _data = _train_data if _split == 'train' else _val_data
    ix = torch.randint(len(_data) - model_parameters['block_size'], (model_parameters['batch_size'],))
    x = torch.stack([_data[i:i+model_parameters['block_size']] for i in ix])
    y = torch.stack([_data[i+1:i+model_parameters['block_size']+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(_model, _eval_iters, _data):
    out = {}
    _model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(_eval_iters)
        for k in range(_eval_iters):
            X, Y = get_batch(split, _data)
            logits, loss = _model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    _model.train()
    return out


def generate_response(_model, _query, max_new_tokens=60):
    _model.eval()
    input_ids = torch.tensor(tokenizer.encode(_query), dtype=torch.long).unsqueeze(0).to(device)
    if model_parameters['with_memory']:
        output_ids = _model.generate_compressed(input_ids, max_new_tokens=max_new_tokens)
    else:
        output_ids = _model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0].tolist())


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


def idle_mode(_model=None):
    print(f"Shakespare Idle Mode")


def train_mode(_model=None):
    print(f"Shakespare Train Mode")


def infer_mode(_model=None):
    print(f"Shakespare infer mode")


def load_main_model(_model, _model_path):
    if os.path.exists(_model_path):
        print(f'Using {_model_path} ')
        _model.load_state_dict(torch.load(_model_path))
        # create a PyTorch optimizer
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=model_parameters['learning_rate_fine'])
        model_parameters['fine_tune'] = True
    else:
        print(f'No model found in {_model_path}, creating new model')
        # create a PyTorch optimizer
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=model_parameters['learning_rate'])
    return _optimizer


def load_preprocessor_model(_model, _model_path):
    if os.path.exists(_model_path):
        print(f'Using {_model_path} ')
        _model.load_state_dict(torch.load(_model_path))
        # create a PyTorch optimizer
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=model_parameters['learning_rate_fine'])
        fine_tune = True
        return _optimizer
    else:
        # create a PyTorch optimizer
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=model_parameters['learning_rate'])
        return _optimizer

# Define a function to generate responses
def generate_response(_model, _query, max_new_tokens=60):
    _model.eval()
    input_ids = torch.tensor(tokenizer.encode(_query), dtype=torch.long).unsqueeze(0).to(device)
    if model_parameters['with_memory']:
        output_ids = _model.generate_compressed(input_ids, max_new_tokens=max_new_tokens) # or here
    else:
        output_ids = _model.generate(input_ids, max_new_tokens=max_new_tokens) #here is an issue
    return tokenizer.decode(output_ids[0].tolist())

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

training_loss = []
validation_loss = []
iteration_times = []

def train(_scheduler, _optimizer, _data, _model):
    _max_iters = model_parameters['max_iters']
    _eval_interval = model_parameters['eval_interval']
    _min_val_loss = model_parameters['min_val_loss']
    _loss_separation = model_parameters['loss_separation']
    _eval_iters = model_parameters['eval_iters']

    scaler = GradScaler()
    timer = TrainingTimer( model_parameters['max_iters'], model_parameters['eval_iters'])
    start_time = time.time()

    for iter in range(_max_iters):
        iter_start_time = time.time()

        # every once in a while evaluate the loss on train and val sets
        if iter % _eval_interval == 0 or iter == _max_iters - 1:
            print("=" * 25)
            losses = estimate_loss(_model, _eval_iters, _data)

            # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            training_loss.append(losses['train'])
            validation_loss.append(losses['val'])

            if losses['val'] < _min_val_loss:
                print(f"Val loss {losses['val']} below {_min_val_loss}, exit train loop")
                break

            separation = losses['val'] - losses['train']
            if separation > _loss_separation :
                print(f"Loss Separation {separation} below {_loss_separation}, exit train loop")
                break

            _scheduler.step(losses['val'])

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
        xb, yb = get_batch('train', _data)

        _optimizer.zero_grad(set_to_none=True)

        # evaluate the loss
        with autocast():
            # xb = xb.requires_grad_(True)
            # yb = yb.requires_grad_(True)
            logits, loss = _model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(_optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        gc.collect()

def createDataset(_datafile, split=0.9):

    with open(_datafile, 'r', encoding='utf-8') as f:
        text = f.read()
    
    rawData = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(split*len(rawData)) # first 90% will be train, rest val
    train_data = rawData[:n]
    val_data = rawData[n:]

    return train_data, val_data

def main():

    model_file = model_parameters['model_file']
    # define the three models
    # preProcessorModel = GPTLanguageModel(params=params)
    mainModel = GPTLanguageModel(params=params)
    # postProcessorModel = GPTLanguageModel(params=params)
    # image processing model
    # sound processing model

    # Additional Characters
    additional_chars = '<>[]{}123456780'

    endcoded = tokenizer.encode(additional_chars)
    print(tokenizer.get_tokens())
    tokenizer.append_special("<math>")
    tokenizer.append_special("</math>")
    print(tokenizer.tokens_numbers)

    optimizer = load_main_model(mainModel, model_parameters['model_file'])    
    mainModel.to(device)

    # mainModel = mainModel.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in mainModel.parameters())/1e6, 'M parameters')

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)
    
    datafile = model_parameters['data_file']

    dataset = createDataset(datafile)

    # train(_scheduler=scheduler,
    #       _optimizer=optimizer, 
    #       _data = dataset, 
    #       _model = mainModel)

    # training works now what....? inferance after training.

    print(f"Saving {model_file}")
    # Save the model
    # torch.save(mainModel.state_dict(), model_file)

        # Example queries
    query = "[Q]: Who is Romeo?\n"
    response = generate_response(mainModel, query)
    print(response)

    pass


if __name__ == "__main__":
    main()
