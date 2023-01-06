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

nn_approximator = NNApproximator(2, 4)


#print("learnable parameters of the neural network:", list(nn_approximator.parameters()))



#@title steepest descent for multi-criteria learning
@dataclass
class SteepestDescent:
    ndim: int
    nu: float
    sigma: float
    eps: float




    """def grad(self, f, x, h=1e-4):
        
        g = torch.zeros_like(x)

        for i in range(self.ndim):
            z = x.clone()  #needed to avoid in-place operation as x required gradient. 
            tmp = z[i]
            z[i] = tmp + h
            yr = f(z)
            z[i] = tmp - h
            yl = f(z)

            g[i] = (yr - yl) / (2 * h)

        print("grad shape: ", g.shape)
        return g"""
    

    def grad(self,f, x):

        gradients  = torch.autograd.grad(f(x), x)[0]

        #print("gradients:", gradients)

        #print("gradients shape:", gradients.shape)
        return gradients



    def nabla_F(self, x):
        obj = ObjNew()
        F = obj.Fss()  # I think this is not necessary. Probably we can do it without Fss and just with Fs
        nabla_F = torch.zeros((len(F), self.ndim)) # (m, n) dimensional matrix
        for i, f in enumerate(F):
            
            #print("jacobian initial:", nabla_F.shape, "jacobian cal:", self.grad(F[i], x).shape)
            
            nabla_F[i] = torch.transpose(self.grad(F[i], x), 0, 1)  #Here I took a transpose compared to the original code. 


            print("gradients", self.grad(F[i], x))


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

    
    def armijo2(self, d, x):
        power = 0
        obj = ObjNew()
        t = pow(self.nu, power)
        Fl = obj.Fs(x + t * d)
        Fr = obj.Fs(x)
        Re = self.sigma * t * torch.matmul(self.nabla_F(x), d)
        #print("my name is junaid")

        #print("checking values of loss functions:", Fl, Fr, Re)

        #print("checking type of loss functions:", type(Fl), type(Fr), type(Re))
        while torch.all(Fl > Fr + Re):
        #while 1 != 0:   
           
            t *= self.nu
            Fl = obj.Fs(x + t * d)
            Fr = obj.Fs(x)

            #print("shapes of tensors:", self.nabla_F(x).shape, d.shape)
            Re = self.sigma * t * torch.matmul(self.nabla_F(x), d)

            print("t inside:", t)

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

        #print("the value of d is", d)


        print(abs(th) > self.eps)

        i = 0
        while abs(th) > self.eps:
            
            i = i + 1

            t = self.armijo2(d, x)

            print("i:", i, "t", t)

            x = x + t * d

            d = minimize(callable_phi, d, method = 'bfgs')   #here i replaced the array with a list
            d = d.x
            th = self.theta(d, x)


        return x
