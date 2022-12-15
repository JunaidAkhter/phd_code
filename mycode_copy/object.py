"""In this example we demonstrate how to use multi-criteria learning  for physics informed neural networks."""

from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchmin import minimize
from typing import Any, Callable
from dataclasses import dataclass

R = 1.0
F0 = 1.0

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
        #print("shape of output layer:", self.layer_out(out).shape)
        return self.layer_out(out)

nn_approximator = NNApproximator(4, 10)


def f() -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    result = lambda x: nn_approximator(x)
    return result

#@title calculating the differential using auto-grad
def df(x, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    #df_value = f(nn, x)
    x = x

    print("the value of x is:", x, "and the type of x is:", type(x))

    #df_value = f(nn)(x) #correct way of obtaining df_value

    print("value", f()(x))

    df_value = f()(x) #correct way of obtaining df_value

    
    #print("printing the output shape", df_value.shape)
    
    
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]


    print("printing the value here", df_value)
    return df_value


# defining the objective functions. 

class ObjNew:

    def f(self, x: torch.Tensor = None, verbose: bool = False) -> torch.float:

        print("printing the types:", type(x), type(R))

        loss =  df(x) - R * x * (1 - x) 

        print("the value of loss is:", loss, "and the type of loss is:", type(loss))


        return loss.pow(2).mean()


    """def g(self, nn: NNApproximator, x:torch.Tensor = None, verbose: bool = False) -> torch.float:
        boundary = torch.Tensor([0.0])
        boundary.requires_grad = True
        boundary_loss = f(nn, boundary) - F0
        return boundary_loss"""
    
    def g(self,x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        """replicating the function fu in order to check if the code is working correctly."""

        loss =  df(x) - R * x * (1 - x) 

        print("the value of loss is:", loss, "and the type of loss is:", type(loss))


        return loss.pow(2).mean()


   # def Fs(self, nn: NNApproximator, x):
   #     return torch.Tensor([self.fu(nn, x), self.g(nn, x)])  # This one does not work and raises the error, 

    """onle one element tensors can be converted to Python scalars"""

        
    def Fs(self, x):
        return [self.f(x), self.g(x)]

    def Fss(self):
        return np.array([self.f, self.g])

      #  return self.fu(x)


#obj = ObjNew()
#print(" objective function fu", obj.fu(x))

