"""Learning a function from a data set : https://deepxde.readthedocs.io/en/latest/demos/function/func.html"""

import deepxde as dde
import numpy as np

def func(x):

    return x * np.sin(5*x)

geom = dde.geometry.Interval(-1, 1)

num_train = 16 
num_test = 100

data = dde.data.Function(geom, func, num_train, num_test)

print("data: ", data)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 3 + [1], activation , initializer)

model = dde.Model(data,  net)
model.compile("adam", lr = 0.001, metrics = ["l2 relative error"])


losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

print(geom)
print("Testing")