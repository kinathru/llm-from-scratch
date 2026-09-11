import torch
from modules.utils import *

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("Embedding layer weights : \n", embedding_layer.weight)
print()
print("Embedding vector for Token ID 3 : \n", embedding_layer(torch.tensor([3])))
print()
print("Embedding vectors for input IDs : \n", embedding_layer(input_ids))

print()

print("------------------- Embedding with a 256-dimensional vector -----------------")
vocab_size = 50257  # Vocab size for the previous novel. I guess 😜
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Each token will be now embedded into a 256-dim vector

raw_text = download_text()
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs : \n", inputs)
print("\nInputs shape: \n", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings shape: \n", token_embeddings.shape)

# Create positional embedding layer (absolute embedding approach)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("\nPos embeddings shape: \n", pos_embeddings.shape)

# Add positional embeddings to token embeddings
input_embeddings = token_embeddings + pos_embeddings
print("\nInput embeddings shape: \n", input_embeddings.shape)