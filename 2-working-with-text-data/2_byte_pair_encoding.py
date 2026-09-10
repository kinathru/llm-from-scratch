from importlib.metadata import version
import tiktoken
from tiktoken import Encoding
from modules.utils import *

print("tiktoken version: ", version("tiktoken"))


def print_encoded_and_decoded_text(text: str, tokenizer: Encoding):
    print("Sample Text  : ", text)
    encoded_ints = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("Encoded Data : ", encoded_ints)
    decoded_text = tokenizer.decode(encoded_ints)
    print("Decoded Text : ", decoded_text)
    print()


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces "
            "of someunknownPlace.")
    print_encoded_and_decoded_text(text, tokenizer)
    text = "Akwirw ier"
    print_encoded_and_decoded_text(text, tokenizer)

    raw_text = download_text()
    enc_text = tokenizer.encode(raw_text)
    print("Length of encoded text : ", len(enc_text))

    enc_sample = enc_text[50:]
    print("Sample of encoded text : ", enc_sample)

    # Array slicing in Python
    # sequence[start:stop:step]
    # start: The starting index of the slice (included). Defaults to 0.
    # stop: The ending index of the slice (excluded). Defaults to the length of the sequence.
    # step: The increment value between each element. Defaults to 1.

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x : {x}")
    print(f"y :      {y}")
    print()

    # Implementing sliding window context -> target
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    print()
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


if __name__ == '__main__':
    main()
