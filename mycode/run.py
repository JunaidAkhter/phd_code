
import numpy as np
from mcml_steepest import SteepestDescent
from object import ObjNew
from model import NNApproximator
import torch


sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = ObjNew()

domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)


nn_approximator = NNApproximator(4, 10)


sd = SteepestDescent(
    ndim=10,    # I think this should be changed to the length of the input vector "x" i.e., len(x)
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

f_opt = sd.steepest(x)
