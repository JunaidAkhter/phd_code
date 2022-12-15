
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchmin import minimize
from typing import Any, Callable
from dataclasses import dataclass



class NNApproximator(nn.Module):

    """Neural network approximator to approximate the solution of the differential equation."""

    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(1, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x):
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)