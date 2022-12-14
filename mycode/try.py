import torch

matrix = torch.zeros((4,3))

print("the shape of matrix is: ", matrix.shape)

row = torch.tensor([0,1,2])

row = row.reshape((3,1))

print(torch.flatten(row))

row = torch.flatten(row)


print("the shape of row is: ", row.shape)

matrix[0] = row 

