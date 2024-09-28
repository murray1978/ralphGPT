class Tokenizer:
    def __init__(self):
        self.tokens_special = [
            "<statement>", "</statement>",
            "<character>", "</character>",
            "<question>", "</question>",
        ]
        self.tokens_numbers = r"1234567890/+-*="
        self.tokens_tokens = r"\<>[]{}!@#%^&()$"
        self.tokens_sentence = r"? ,.:;'\""
        self.tokens_alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\t\n"
        self.chars = sorted(list(
            set(self.tokens_alpha)
            | set(self.tokens_numbers)
            | set(self.tokens_sentence)
            | set(self.tokens_tokens)
        ))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, _text):
        _tokens = []
        i = 0
        while i < len(_text):
            match = False
            for _token in self.tokens_special:
                if _text[i:i+len(_token)] == _token:
                    _tokens.append(self.stoi[_token])
                    i += len(_token)
                    match = True
                    break
            if not match:
                _tokens.append(self.stoi[_text[i]])
                i += 1
        return _tokens

    def decode(self, data):
        return ''.join([self.itos[i] for i in data])

    def append_special(self, _token):
        if self.tokens_special.count(_token) == 0:
            self.tokens_special.append(_token)

    def get_tokens(self):
        _tokens = list( set(self.tokens_special)
                        | set(self.tokens_alpha)
                        | set(self.tokens_tokens)
                        | set(self.tokens_sentence)
                        | set(self.tokens_numbers)
                        )
        return _tokens

    def get_vocab_size(self):
        return self.vocab_size
