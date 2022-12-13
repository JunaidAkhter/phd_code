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


"""def f(nn: NNApproximator, x: torch.Tensor) -> torch.Tensor:

    result = lambda x: nn(x)
    return result"""

def f(nn: NNApproximator) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""

    result = lambda x: nn(x)
    return result

#@title check
domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)

nn_approximator = NNApproximator(4, 10)
f(NNApproximator(4, 10))(x)

#@title calculating the differential using auto-grad
def df(nn: NNApproximator, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    #df_value = f(nn, x)

    print("value of df:", f(nn)(x))


    df_value = f(nn)(x) #correct way of obtaining df_value

    
    #print("printing the output shape", df_value.shape)
    
    
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


class Obj:

    def f(self, x):
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x):
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x):
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])



# defining the objective functions. 

class ObjNew:

    def fu(self,nn: NNApproximator, x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        return  df(nn, x) - R * x * (1 - x) 

    """def g(self, nn: NNApproximator, x:torch.Tensor = None, verbose: bool = False) -> torch.float:
        boundary = torch.Tensor([0.0])
        boundary.requires_grad = True
        boundary_loss = f(nn, boundary) - F0
        return boundary_loss"""
    
    def g(self,nn: NNApproximator, x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        """replicating the function fu in order to check if the code is working correctly."""

        
        return  df(nn, x) - R * x * (1 - x) 

   # def Fs(self, nn: NNApproximator, x):
   #     return torch.Tensor([self.fu(nn, x), self.g(nn, x)])  # This one does not work and raises the error, 

    """onle one element tensors can be converted to Python scalars"""

        
    def Fs(self, nn: NNApproximator, x):
        return [self.fu(nn, x), self.g(nn, x)]

    def Fss(self):
        return np.array([self.fu, self.g])

      #  return self.fu(x)



#@title steepest descent for multi-criteria learning
@dataclass
class SteepestDescent:
    ndim: int
    nu: float
    sigma: float
    eps: float

    def grad(self, f, x, h=1e-4):
        
        g = torch.zeros_like(x)
        for i in range(self.ndim):
            z = x.clone()  #needed to avoid in-place operation as x required gradient. 
            tmp = z[i]
            z[i] = tmp + h
            yr = f(z)
            z[i] = tmp - h
            yl = f(z)
            g[i] = (yr - yl) / (2 * h)
            #x[i] = tmp
        #print("this is how g looks like:", g.shape)
        return g

    def nabla_F(self, x):
        obj = ObjNew()
        F = obj.Fss()
        nabla_F = torch.zeros((len(F), self.ndim)) # (m, n) dimensional matrix
        for i, f in enumerate(F):
            
            #print("jacobian initial:", nabla_F.shape, "jacobian cal:", self.grad(F[i], x).shape)
            
            nabla_F[i] = torch.transpose(self.grad(F[i], x), 0, 1)  #Here I took a transpose compared to the original code. 
        return nabla_F

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(torch.matmul(nabla_F, d)) + 0.5 * torch.norm(d) ** 2

        
    def create_callable_phi(self, x):

        nabla_F = self.nabla_F(x)

        callable_phi = lambda d: max(torch.matmul(nabla_F, d)) + 0.5 * torch.norm(d) ** 2
        return callable_phi

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * torch.norm(d) ** 2

    def armijo(self, d, x):
        power = 0
        obj = ObjNew()
        t = pow(self.nu, power)
        Fl = torch.tensor(obj.Fs(x + t * d))
        Fr = torch.tensor(obj.Fs(x))
        Re = self.sigma * t * torch.matmul(self.nabla_F(x), d)
        while torch.all(Fl > Fr + Re):
            t *= self.nu
            Fl = torch.tensor(obj.Fs(x + t * d))
            Fr = torch.tensor(obj.Fs(x))
            Re = self.sigma * t * torch.matmul(self.nabla_F(x), d)
        return t

    
    def armijo2(self, d, x, nn_approximator):
        power = 0
        obj = ObjNew()
        t = pow(self.nu, power)
        Fl = obj.Fs(nn_approximator, x + t * d)
        Fr = obj.Fs(nn_approximator, x)
        Re = self.sigma * t * torch.matmul(self.nabla_F(x), d)
        while np.all(Fl > Fr + Re):
            t *= self.nu
            Fl = obj.Fs(nn_approximator, x + t * d)
            Fr = obj.Fs(nn_approximator, x)
            Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        return t
    
    def steepest(self, x, nn_approximator):

        #d = [torch.fmin(self.phi, x, args=(x, ))] #here i replaced the array with a list

        # function to be minimized has to be calleble
        callable_phi = self.create_callable_phi(x)
        # starting point for optimization
        d0 = torch.ones_like(x)
        
        d = minimize(callable_phi, d0, method = 'cg') #here i replaced the array with a list
        # only get optimal calue
        d = d.x
        th = self.theta(d, x)
        while abs(th) > self.eps:
            t = self.armijo2(d, x, nn_approximator)
            x = x + t * d
            d = minimize(callable_phi, d0, method = 'bfgs')   #here i replaced the array with a list
            d = d.x
            th = self.theta(d, x)
        return x


obj = ObjNew()
domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)

nn_approximator = NNApproximator(4, 10)

print(" objective function fu", obj.fu(nn_approximator,x))

obj.Fs(nn_approximator, x)

sd = SteepestDescent(
    ndim=10,    # I think this should be changed to the length of the input vector "x" i.e., len(x)
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

f_opt = sd.steepest(x, nn_approximator)
print(obj.Fs(f_opt)) # Pareto optimal