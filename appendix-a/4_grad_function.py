import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])  # True Label
x1 = torch.tensor([1.1])  # input feature
w1 = torch.tensor(2.2, requires_grad=True)  # Weight
b = torch.tensor([0.0], requires_grad=True)  # Bias unit
z = x1 * w1 + b  # Calculation of net input
a = torch.sigmoid(z)  # Activation/prediction
loss = F.binary_cross_entropy(a, y)  # Calculate loss against predicted value

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(f"Net Input (z) = {z}")
print(f"Activation function output (a) = {a}")
print(f"Calculated loss = {loss}")
print(f"Loss Gradients L_w1={grad_L_w1}, L_b={grad_L_b}")

# -----------------------------------------------------
#           Automatic Grad Calculation
# -----------------------------------------------------
print("\nAutomatic Grad Calculation\n")
loss.backward()
print(f"L_w1={w1.grad}")
print(f"L_b={b.grad}")
