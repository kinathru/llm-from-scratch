from modules.utils import *
from modules.simple_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


def main():
    text = download_text()
    # regex_tokenize("Hello, World. This, is a test. Is this-- a test?")
    pre_processed = regex_tokenize(text)
    print(len(pre_processed))
    print(pre_processed[:30])

    # Build vocabulary
    vocab = build_vocabulary(pre_processed)

    # Print first 10 items of the vocabulary
    print(f"Printing first 10 items of the vocabulary")
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 10:
            break

    # Print last 5 items of the vocabulary
    print(f"Printing last 5 items of the vocabulary")
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    tokenizer = SimpleTokenizerV1(vocab=vocab)
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    try:
        text = "Hello, do you like tea?"
        print(tokenizer.encode(text))
    except Exception as e:
        print(e)

    tokenizer = SimpleTokenizerV2(vocab=vocab)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))

    print(text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

if __name__ == '__main__':
    main()
