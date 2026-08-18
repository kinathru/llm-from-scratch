import torch

tensor0d = torch.tensor(1)
print(tensor0d)
print(f"Data type of tensor0d {tensor0d.dtype}")

tensor1d = torch.tensor([1, 2, 3])
print(tensor1d)

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d)

tensor3d = torch.tensor(
    [
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [10, 20]
        ]
    ])
print(tensor3d)
print(f"Data type of tensor3d {tensor3d.dtype}")

float_vec = torch.tensor([10.25, 3.48, 8.47])
print(float_vec)
print(f"Data type of float_vec {float_vec.dtype}")

to_float32_vector = tensor2d.to(torch.float32)
print(to_float32_vector)
print(f"Data type of to_float32_vector {to_float32_vector.dtype}")
print(f"Shape of to_float32_vector {to_float32_vector.shape}")

reshaped_tensor = tensor2d.reshape(3, 2)
print(f"Reshaped tensor to (3,2) : ", reshaped_tensor)

print(f"View reshaped tensor to (3,2) : ", tensor2d.view(3, 2))
print(f"Transposed 2D Tensor : ", tensor2d.T)

multiplied_2d_tensors = tensor2d.matmul(tensor2d.T)
print(f"Multiplied 2D Tensor : ", multiplied_2d_tensors)
print(f"Multiplied 2D Tensor using @: ", tensor2d @ tensor2d.T)