"""In this example we demonstrate how to use multi-criteria learning  for physics informed neural networks."""

from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchmin import minimize
from typing import Any, Callable
from dataclasses import dataclass
from object import ObjNew
from model import NNApproximator

nn_approximator = NNApproximator(4, 10)

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

    
    def armijo2(self, d, x):
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
    
    def steepest(self, x):

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
