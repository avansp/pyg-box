import torch

class SimpleDenseNet(torch.nn.Module):
    """A simple DenseNet classifier torch module with 4 layers."""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lin1_size: int = 56,
        lin2_size: int = 28,
        lin3_size: int = 14
    ) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, lin1_size),
            torch.nn.BatchNorm1d(lin1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(lin1_size, lin2_size),
            torch.nn.BatchNorm1d(lin2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(lin2_size, lin3_size),
            torch.nn.BatchNorm1d(lin3_size),
            torch.nn.ReLU(),
            torch.nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        sz = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(sz[0], -1)

        return self.model(x)

        
