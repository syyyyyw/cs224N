import torch
device=torch.device('cuda')
a=torch.tensor(1).to(device)
print(a)