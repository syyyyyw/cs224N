
import torch
import torch.nn as nn
conv1=nn.Conv1d(in_channels=256,out_channels=100,kernel_size=2)
input=torch.randn(32,32,35,256)
input=input.permute(0,1,3,2)
output=conv1(input)
print(output.shape)
# x=torch.randn(10,10,10)
# m=nn.MaxPool1d(10)
# x=m(x).squeeze(-1)
# print(x.shape)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.maxpool=nn.MaxPool1d(20)   #maxpool是没有参数的 所以写forward里面也没有问题的
#         self.conv1 = nn.Conv1d(1, 6, 3)
#     def forward(self,x):
#         return x
#
#
# net=Net()
# for parameters in net.parameters():
#     print(parameters)
# print('-'*80)
# for name,parameters in net.named_parameters():
#     print(name,':',parameters.size())
