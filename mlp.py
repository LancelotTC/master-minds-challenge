import torch
import torch.nn as nn
from typing import Any
from utils import Monad
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return Monad(x).bind(self.fc1).bind(F.relu).bind(self.fc2).value


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    IN_FEATURES: int = ...
    HIDDEN_LAYERS: int = 3

    model = MLP(IN_FEATURES, HIDDEN_LAYERS, 1)  # output size of 1 for raw continuous output logits (regression)
    model.to(device)
