from typing import Callable

import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from torch import nn
import numpy as np

R = 1.0
F0 = 1.0

class NNApproximator(nn.Module):
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


def f(nn: NNApproximator, x: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return nn(x)

#@title check
domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)

nn_approximator = NNApproximator(4, 10)
f(NNApproximator(4, 10), x)

#@title calculating the differential using auto-grad
def df(nn: NNApproximator, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = f(nn, x)

    
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

#@title calculating the full loss function
def compute_loss(
    nn: NNApproximator, x: torch.Tensor = None, verbose: bool = False
) -> torch.float:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    interior_loss = df(nn, x) - R * x * (1 - x)

    boundary = torch.Tensor([0.0])
    boundary.requires_grad = True
    boundary_loss = f(nn, boundary) - F0
    final_loss = interior_loss.pow(2).mean() + boundary_loss ** 2
    return final_loss

#@title training the model 
def train_model(
    nn: NNApproximator,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000,
) -> NNApproximator:

    loss_evolution = []

    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate)
    # if we want to use a different optimization techniques then the above
    # line has to be changed. 


    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

            loss_evolution.append(loss.detach().numpy())

        except KeyboardInterrupt:
            break

    return nn, np.array(loss_evolution)

def check_gradient(nn: NNApproximator, x: torch.Tensor = None) -> bool:

    eps = 1e-4
    dfdx_fd = (f(nn, x + eps) - f(nn, x - eps)) / (2 * eps)
    dfdx_sample = df(nn, x, order=1)

    return torch.allclose(dfdx_fd.T, dfdx_sample.T, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    from functools import partial

    domain = [0.0, 1.0]
    x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
    x = x.reshape(x.shape[0], 1)

    nn_approximator = NNApproximator(4 , 10)
    assert check_gradient(nn_approximator, x)

    # f_initial = f(nn_approximator, x)
    # ax.plot(x.detach().numpy(), f_initial.detach().numpy(), label="Initial NN solution")

    # train the PINN
    loss_fn = partial(compute_loss, x=x, verbose=True)
    nn_approximator_trained, loss_evolution = train_model(
        nn_approximator, loss_fn=loss_fn, learning_rate=0.1, max_epochs=10_000
    )

    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)

    # numeric solution
    def logistic_eq_fn(x, y):
        return R * x * (1 - x)

    numeric_solution = solve_ivp(
        logistic_eq_fn, domain, [F0], t_eval=x_eval.squeeze().detach().numpy()
    )

    # plotting
    fig, ax = plt.subplots()

    f_final_training = f(nn_approximator_trained, x)
    f_final = f(nn_approximator_trained, x_eval)

    ax.scatter(x.detach().numpy(), f_final_training.detach().numpy(), label="Training points", color="red")
    ax.plot(x_eval.detach().numpy(), f_final.detach().numpy(), label="NN final solution")
    ax.plot(
        x_eval.detach().numpy(),
        numeric_solution.y.T,
        label=f"Analytic solution",
        color="green",
        alpha=0.75,
    )
    ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
    ax.legend()

    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    plt.show()

#@title Here we define the objective functions that we want to optimize. 

import numpy as np
#from nptyping import NDArray   # I am not using it as it was giving some syntax errors. I still have to figure out how ot use it  properly 
from typing import Any, Callable

class Obj:

    def f(self, x):
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x):
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x):
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])


import numpy as np
from scipy.optimize import fmin
#from nptyping import NDArray
from typing import Any, Callable
from dataclasses import dataclass
#from obj_func import Obj

@dataclass
class SteepestDescent:
    ndim: int
    nu: float
    sigma: float
    eps: float

    def grad(self, f, x, h=1e-4):
        g = np.zeros_like(x)
        for i in range(self.ndim):
            tmp = x[i]
            x[i] = tmp + h
            yr = f(x)
            x[i] = tmp - h
            yl = f(x)
            g[i] = (yr - yl) / (2 * h)
            x[i] = tmp

        #print("shape of g:", g.shape)

        return g

    def nabla_F(self, x):
        obj = Obj()
        F = obj.Fss()    # why is Fss here and not Fs
        nabla_F = np.zeros((len(F), self.ndim)) # (m, n) dimensional matrix
        #print("shape if jacobian initial:", nabla_F.shape)
        for i, f in enumerate(F):
            nabla_F[i] = self.grad(F[i], x)
        return nabla_F

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(np.dot(nabla_F, d)) + 0.5 * np.linalg.norm(d) ** 2

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * np.linalg.norm(d) ** 2

    def armijo(self, d, x):
        power = 0
        obj = Obj()
        t = pow(self.nu, power)
        Fl = np.array(obj.Fs(x + t * d))
        Fr = np.array(obj.Fs(x))
        Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        while np.all(Fl > Fr + Re):
            t *= self.nu
            Fl = np.array(obj.Fs(x + t * d))
            Fr = np.array(obj.Fs(x))
            Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        return t
    
    def steepest(self, x):
        d = np.array(fmin(self.phi, x, args=(x, )))
        th = self.theta(d, x)
        while abs(th) > self.eps:
            t = self.armijo(d, x)
            x = x + t * d
            d = np.array(fmin(self.phi, x, args=(x, )))
            th = self.theta(d, x)
        return x

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

x_init = np.array([1, 2])
f_opt = sd.steepest(x_init)
print(obj.Fs(f_opt)) # Pareto optimal

import matplotlib.pyplot as plt 

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

np.random.seed(10)

nmax = 100 # the number of computing the Pareto optimal
x_opt = np.zeros((nmax, 2))
f_opt = np.zeros((nmax, 2))

for i in range(nmax):
    x_init = np.random.rand(2)
    x = sd.steepest(x_init)
    x_opt[i] = x
    f_opt[i] = obj.Fs(x)

fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[0].scatter(f_opt[:, 0], f_opt[:, 1], c="r", alpha=0.7)
ax[1].scatter(x_opt[:, 0], x_opt[:, 1], c="r", alpha=0.7)

ax[0].set_xlabel("$f_1(x)$")
ax[0].set_ylabel("$f_2(x)$")
ax[1].set_xlabel("$x_1$")
ax[1].set_ylabel("$x_2$")
ax[0].set_title("The Pareto Front")
ax[1].set_title("The Pareto Optimal set")

fig.tight_layout()
plt.show()

from torchmin import minimize

#@title steepest descent adapted to pytorch.

import numpy as np
from scipy.optimize import fmin
#from nptyping import NDArray
from typing import Any, Callable
from dataclasses import dataclass
#from obj_func import Obj

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
        obj = Obj()
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
        obj = Obj()
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

#@title Here we define the objective functions that we want to optimize. 

import numpy as np
#from nptyping import NDArray
from typing import Any, Callable

class ObjNew:

    def fu(self,nn: NNApproximator, x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        return  df(nn, x) - R * x * (1 - x) 

    def g(self, nn: NNApproximator, x:torch.Tensor = None, verbose: bool = False) -> torch.float:
        boundary = torch.Tensor([0.0])
        boundary.requires_grad = True
        boundary_loss = f(nn, boundary) - F0
        return boundary_loss
        
   # def Fs(self, nn: NNApproximator, x):
   #     return torch.Tensor([self.fu(nn, x), self.g(nn, x)])  # This one does not work and raises the error, 

    """onle one element tensors can be converted to Python scalars"""

        
    def Fs(self, nn: NNApproximator, x):
        return [self.fu(nn, x), self.g(nn, x)]

    def Fss(self):
        return np.array([self.fu, self.g])

      #  return self.fu(x)

obj = ObjNew()
domain = [0.0, 1.0]
x = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True)
x = x.reshape(x.shape[0], 1)

nn_approximator = NNApproximator(4, 10)
obj.fu(nn_approximator,x)

obj.Fs(nn_approximator, x)

sd = SteepestDescent(
    ndim=10,    # I think this should be changed to the length of the input vector "x" i.e., len(x)
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

f_opt = sd.steepest(x, nn_approximator)
print(obj.Fs(f_opt)) # Pareto optimal