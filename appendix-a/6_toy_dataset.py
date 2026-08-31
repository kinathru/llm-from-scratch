import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


def main():
    torch.manual_seed(123)  # To use the same shuffling order

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

    test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx + 1} : ", x, y)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    # Stochastic Gradient Descent optimizer with a learning rate of 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters : ", num_params)

    num_epochs = 5

    # One training "epoch" is like one iteration over the full dataset
    for epoch in range(num_epochs):

        # Set the module in training mode.
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
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

            ### LOGGING
            print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train Loss: {loss:.2f}")

        # Set the model in the evaluation mode
        model.eval()

    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
    print(outputs)

    # Make the outputs more legible
    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(outputs, dim=1)
    print(probas)

    predictions = torch.argmax(probas, dim=1)
    print(predictions)

    # Apply argmax directly on outputs insteadof using softmax
    predictions_direct = torch.argmax(outputs, dim=1)
    print(predictions_direct)

    print('Check if predictions are correct : ', predictions == y_train)
    print('Correct predictions count : ', torch.sum(predictions == y_train))

    print('Accuracy (Train) : ', compute_accuracy(model, train_loader))
    print('Accuracy (Test)  : ', compute_accuracy(model, test_loader))

    print('Saving the model to a file')
    torch.save(model.state_dict(), "model.pth")

    print('Loading the model from a file')
    loaded_model = NeuralNetwork(num_inputs=2, num_outputs=2)
    loaded_model.load_state_dict(torch.load("model.pth"))

    loaded_model_params = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
    print("Total number of loaded_model parameters : ", loaded_model_params)




if __name__ == '__main__':
    main()
