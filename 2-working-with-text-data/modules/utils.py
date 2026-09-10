import urllib.request
import re

from torch.utils.data import DataLoader
from .gpt_data_set import GPTDataSetV1

import tiktoken


def download_text() -> str:
    url = ("https://raw.githubusercontent.com/kinathru/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])
    return raw_text


def regex_tokenize(text: str) -> list:
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

    # Remove whitespaces
    result = [item for item in result if item.strip()]

    return result


def build_vocabulary(tokens: list[str]) -> dict[str, int]:
    all_tokens = sorted(set(tokens))

    # Add special tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}

    print(f"Vocabulary size : {len(vocab.items())}")
    return vocab


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
