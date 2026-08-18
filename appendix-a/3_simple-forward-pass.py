import torch
import torch.nn.functional as F

y = torch.tensor([1.0]) # True Label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor(2.2) # Weight
b = torch.tensor([0.0]) # Bias unit
z = x1 * w1 + b # Calculation of net input
a = torch.sigmoid(z) # Activation
loss = F.binary_cross_entropy(a, y)

print(f"Net Input (z) = {z}")
print(f"Activation function output (a) = {a}")
print(f"Calculated loss = {loss}")