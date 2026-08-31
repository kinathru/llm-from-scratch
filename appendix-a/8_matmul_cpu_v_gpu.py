import time

import torch


def matmul_with_time(t1, t2):
    # Start the timer
    start_time = time.perf_counter()

    t3 = torch.matmul(t1, t2.T)
    print("Tensor 01 x Tensor 02 : ", t3)

    # End the timer
    end_time = time.perf_counter()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Time taken: {execution_time:.6f} seconds")


def main():
    torch.manual_seed(123)
    size = (3000, 3000)
    tensor_1 = torch.rand(size=size)
    tensor_2 = torch.rand(size=size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_1_gpu = tensor_1.to(device)
    tensor_2_gpu = tensor_2.to(device)

    matmul_with_time(tensor_1, tensor_2)
    print()
    matmul_with_time(tensor_1_gpu, tensor_2_gpu)

if __name__ == '__main__':
    main()