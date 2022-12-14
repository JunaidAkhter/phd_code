import numpy as np
import matplotlib.pyplot as plt
from mcml_steepest import SteepestDescent
from object import ObjNew
import torch


input_dim = 10 #dimension of the input space


sd = SteepestDescent(
    ndim=input_dim,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)


obj = ObjNew()


#np.random.seed(10)

torch.random.manual_seed(10)

nmax = 100 # the number of computing the Pareto optimal

x_opt = torch.zeros((nmax, input_dim))
f_opt = torch.zeros((nmax, 2))

for i in range(nmax):
    x_init = torch.rand(input_dim, requires_grad=True)

    # reshaping the input. I think this does not need to be done for each iteration
    x_init = x_init.reshape(x_init.shape[0], 1)


    #print("the shape of x_init", x_init.shape)
    x = sd.steepest(x_init)
    
    #print("the shape of x_opt", x_opt.shape)

    #print("the shape of x in visualization file", x.shape)
    
    x_opt[i] = torch.flatten(x)


    f_opt[i] = torch.tensor(obj.Fs(x))



x_opt = x_opt.detach().numpy()
f_opt = f_opt.detach().numpy()



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