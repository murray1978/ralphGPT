# Define your text and additional special tokens
text = "We are accounted poor citizens..."
special_tokens = ["<statement>", "</statement>",
                  "<character>", "</character>",
                  "<question>",  "</question>",
                  ]

# Create a set of unique characters in the text and the special tokens
chars = sorted(list(set(text)))
vocab_size = len(chars) + len(special_tokens)

# Create a mapping from characters and special tokens to integers
stoi = {ch: i for i, ch in enumerate(chars)}
stoi.update({token: i + len(stoi) for i, token in enumerate(special_tokens)})

# Reverse mapping from integers to characters/tokens
itos = {i: ch for ch, i in stoi.items()}


# Update encode function to handle special tokens
def encode(s):
    tokens = []
    i = 0
    while i < len(s):
        match = False
        for token in special_tokens:
            if s[i:i+len(token)] == token:
                tokens.append(stoi[token])
                i += len(token)
                match = True
                break
        if not match:
            tokens.append(stoi[s[i]])
            i += 1
    return tokens


# Update decode function to handle special tokens
def decode(l):
    return ''.join([itos[i] for i in l])


# Example usage
encoded_text = encode("<statement>" + text + "</statement>")
decoded_text = decode(encoded_text)

print("Encoded:", encoded_text)
print("Decoded:", decoded_text)
