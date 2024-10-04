# processing and creating datasets.
# looking at dailyDialog and how to process it for ralphGPT or gptnv.py
#import pandas as pd 
from tokeniser import Tokenizer

tokenizer = Tokenizer()
tokenizer.append_special("__eou__")
encode = tokenizer.encode
decode = tokenizer.decode


