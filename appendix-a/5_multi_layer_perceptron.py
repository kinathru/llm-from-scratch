import torch.nn


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


if __name__ == '__main__':
    torch.manual_seed(123)  # Keep the random number initialization re-producible
    model = NeuralNetwork(50, 3)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters : ", num_params)

    print("Layer 0 weights are :\n", model.layers[0].weight)
    print("Layer 0 weights shape is :\n", model.layers[0].weight.shape)

    print("Layer 0 bias vector is :\n", model.layers[0].bias)
    print("Layer 0 bias vector shape is :\n", model.layers[0].bias.shape)

    X = torch.rand((1, 50))
    out = model(X)
    print(out)

    with torch.no_grad():
        out = torch.softmax(model(X), dim=1)
        print("No gradient output with softmax : ", out)
