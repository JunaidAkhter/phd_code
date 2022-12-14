
import numpy as np
from mcml_steepest import SteepestDescent
from object import ObjNew, NNApproximator
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

print(" objective function f", obj.f(x))

print(" objective function g", obj.g(x))


print(" list of objective functions:", obj.Fs(x))
print("list of objective functions without values", obj.Fss())

sd = SteepestDescent(
    ndim=10,    # I think this should be changed to the length of the input vector "x" i.e., len(x)
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

f_opt = sd.steepest(x)
print("This is the pareto optimal value", obj.Fs(f_opt)) # Pareto optimal

print("the shape of x:", x.shape)