"""In this example we demonstrate how to use multi-criteria learning  for physics informed neural networks."""

from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchmin import minimize
from typing import Any, Callable
from dataclasses import dataclass
from model import NNApproximator

R = 1.0
F0 = 1.0

nn_approximator = NNApproximator(4, 10)

def f() -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    result = lambda x: nn_approximator(x)
    return result

def df(x, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""

    df_value = f()(x) #correct way of obtaining df_value

    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


# defining the objective functions. 

class ObjNew:

    def f(self, x: torch.Tensor = None, verbose: bool = False) -> torch.float:

        loss =  df(x) - R * x * (1 - x) 

        return loss.pow(2).mean()


    """def g(self, nn: NNApproximator, x:torch.Tensor = None, verbose: bool = False) -> torch.float:
        boundary = torch.Tensor([0.0])
        boundary.requires_grad = True
        boundary_loss = f(nn, boundary) - F0
        return boundary_loss"""
    
    def g(self,x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        """replicating the function fu in order to check if the code is working correctly."""

        loss =  df(x) - R * x * (1 - x) 

        return loss.pow(2).mean()


   # def Fs(self, nn: NNApproximator, x):
   #     return torch.Tensor([self.fu(nn, x), self.g(nn, x)])  # This one does not work and raises the error, 

    """onle one element tensors can be converted to Python scalars"""

        
    def Fs(self, x):
        return torch.tensor([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])

      #  return self.fu(x)
