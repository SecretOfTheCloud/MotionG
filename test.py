import torch
import numpy as np

# model = torch.load("./model.pth")
# model.eval()

# # x=torch.tensor(np.zeros([1,316]).astype(np.float32))
# # p=torch.tensor(np.zeros([1,12]).astype(np.float32))
# # z=torch.tensor(np.zeros([1,2]).astype(np.float32))
# x=torch.tensor(np.zeros([1,316]).astype(np.float32))
# p=torch.tensor(np.zeros([1,12]).astype(np.float32))
# z=torch.tensor(np.zeros([1,2]).astype(np.float32))



# y=model(x,p,z)
# print(y)


a=np.array([[1,2,3],[4,5,6]])

np.savetxt('./text.txt', np.c_[a],
 fmt='%d',delimiter='\t')