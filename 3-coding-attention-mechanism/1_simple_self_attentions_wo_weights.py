# Simple self-attention without trainable weights

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts (x^3)
     [0.22, 0.58, 0.33],  # with (x^4)
     [0.77, 0.25, 0.10],  # one (x^5)
     [0.05, 0.80, 0.55]]  # step (x^6)
)

# Calculating attention scores -> dot product between the query and all input tokens
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
print("Empty Scores Tensor      : ", attn_scores_2)
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("Attention Scores Tensor : ", attn_scores_2)

# Dot product explanation
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print("Manual dot product calculated        : ", res)
print("Dot product calculated with torch    : ", torch.dot(inputs[0], query))

# Normalizing attention scores to obtain attention weights
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention Weights    : ", attn_weights_2_tmp)
print("Sum                  : ", attn_weights_2_tmp.sum())


# Normalizing attention scores using softmax
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention Weights    : ", attn_weights_2_naive)
print("Sum                  : ", attn_weights_2_naive.sum())

# Normalizing attention weights using PyTorch SoftMax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention Weights    : ", attn_weights_2)
print("Sum                  : ", attn_weights_2.sum())

# Calculating the context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    print(" > Index                 : ", i)
    print(" > Attention Weight      : ", attn_weights_2[i])
    print(" > Input Embedding       : ", x_i)
    print(" > Context Vector Int    : ", attn_weights_2[i] * x_i)
    context_vec_2 += attn_weights_2[i] * x_i
    print(" > Cumulative Context Vector : ", context_vec_2)
print("Context Vector 2     : ", context_vec_2)

# Calculating attention scores between each input
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("Attention Scores Tensor (For Loop)   : \n", attn_scores)

# Calculating attention scores using matrix multiplication
attn_scores = inputs @ inputs.T
print("Attention Scores Tensor (Matrix Mul) : \n", attn_scores)

# Calculating attention weights by normalizing attention scores
# dim=-1 to normalize across the columns, so that values in each row sums up to 1 (Read the bok)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention Weights                    : \n", attn_weights)

# Validate row sum
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum    : ", row_2_sum)
print("All row sums : ", attn_weights.sum(dim=-1))

# Calculating Context Vectors
all_context_vecs = attn_weights @ inputs
print("All context vectors  : \n", all_context_vecs)
print("Context Vector 2     : ", context_vec_2)