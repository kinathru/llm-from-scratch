import os
import torch.multiprocessing as mp
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn.functional as F

torch.manual_seed(123)  # To use the same shuffling order


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def prepare_dataset(train_ds, test_ds) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(test_ds)
    )
    return train_loader, test_loader


class NeuralNetwork(torch.nn.Module):
    # Coding the number of inputs and outputs as variables allows us to reuse the same code for datasets
    # with different numbers of features and classes
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer - The Linear layer takes the number of input and output nodes as arguments.
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer - Nonlinear activation functions are placed between the hidden layers.
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer - The number of output nodes of one hidden layer has to match the number of inputs of the next layer.
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        # The outputs of the last layer are called logits.
        logits = self.layers(x)
        return logits


def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = (labels == predictions)
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


def main(rank, world_size, num_epochs, train_ds, test_ds):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset(train_ds, test_ds)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)

            # Forward pass
            logits = model(features)

            # Calculate loss
            loss = F.cross_entropy(logits, labels)

            # Sets the gradients from the previous round to 0 to prevent unintended gradient accumulation
            optimizer.zero_grad()

            # Computes the gradients of the loss given the model parameters
            loss.backward()

            # The optimizer uses the gradients to update the model parameters
            optimizer.step()

            print(f"[GPU{rank}] Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                  f" | Batch Size {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

        model.eval()
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Train Accuracy: {train_acc:.2f}")
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test Accuracy: {test_acc:.2f}")
        destroy_process_group()


if __name__ == '__main__':
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6]
    ])

    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True  # Drop the last batch to avoid disturbance in the convergence during training
    )

    print("Number of GPUs available: ", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs, train_ds, test_ds ), nprocs=world_size)
