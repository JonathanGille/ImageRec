import torch

t1 = torch.zeros(4)
t2 = torch.ones(4)
t3 = torch.tensor([1,2,3,4])
t = torch.stack([t1,t2,t3])
tf = torch.flip(t, dims=[0])
print(t)
print(tf)