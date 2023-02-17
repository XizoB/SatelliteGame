import torch

tensor_0 = torch.arange(0, 20).view(4, 5)
print(tensor_0,tensor_0.shape)

index = torch.tensor([[0,1,2,3]]).view(-1,1)
print(index,index.shape)
tensor_1 = tensor_0.gather(1, index)  # 0是index里面的数代表行， 1是index里面的数代表列
print(tensor_1)