import torch

t1 = torch.arange(1,17,1).reshape((1,1,4,4))
t2 = torch.arange(1,17,1).reshape((1,1,4,4))

print(t1)

t1 = t1.transpose(2,3)

print(t1)

print(torch.matmul(t1,t2))