import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, xDim, yDim, hidden, dropout):
        super(Model, self).__init__()
        self.w00 = nn.Linear(xDim, hidden)
        self.w01 = nn.Linear(xDim, hidden)
        self.w02 = nn.Linear(xDim, hidden)
        self.w03 = nn.Linear(xDim, hidden)
        self.w04 = nn.Linear(xDim, hidden)
        self.w05 = nn.Linear(xDim, hidden)
        self.w06 = nn.Linear(xDim, hidden)
        self.w07 = nn.Linear(xDim, hidden)

        self.w10 = nn.Linear(hidden, hidden)
        self.w11 = nn.Linear(hidden, hidden)
        self.w12 = nn.Linear(hidden, hidden)
        self.w13 = nn.Linear(hidden, hidden)
        self.w14 = nn.Linear(hidden, hidden)
        self.w15 = nn.Linear(hidden, hidden)
        self.w16 = nn.Linear(hidden, hidden)
        self.w17 = nn.Linear(hidden, hidden)

        self.w20 = nn.Linear(hidden, yDim)
        self.w21 = nn.Linear(hidden, yDim)
        self.w22 = nn.Linear(hidden, yDim)
        self.w23 = nn.Linear(hidden, yDim)
        self.w24 = nn.Linear(hidden, yDim)
        self.w25 = nn.Linear(hidden, yDim)
        self.w26 = nn.Linear(hidden, yDim)
        self.w27 = nn.Linear(hidden, yDim)

        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation0 = nn.ELU(inplace = True)
        self.activation1 = nn.ELU(inplace = True)

        self.Gating1 = nn.Linear(12, 32)
        self.Gating2 = nn.Linear(32, 32)
        self.Gating3 = nn.Linear(32, 8)

    def forward(self, x, p, z):
        #z 6+6
        G1 = self.Gating1(p)
        G2 = self.Gating2(G1)
        G3 = self.Gating3(G2)

        hidden = self.dropout0(x)
        # print(self.w00(x).shape)
        # print(G3.shape)

        hidden = self.w00(x) * G3[:,0:1] + self.w01(x) * G3[:,1:2] \
            + self.w02(x) * G3[:,2:3] + self.w03(x) * G3[:,3:4] \
            + self.w04(x) * G3[:,4:5] + self.w05(x) * G3[:,5:6] \
            + self.w06(x) * G3[:,6:7] + self.w07(x) * G3[:,7:8] 
            
        hidden = self.dropout1(self.activation0(hidden))
        hidden = self.w10(hidden) * G3[:,0:1] + self.w11(hidden) * G3[:,1:2] \
            + self.w12(hidden) * G3[:,2:3] + self.w13(hidden) * G3[:,3:4] \
            + self.w14(hidden) * G3[:,4:5] + self.w15(hidden) * G3[:,5:6] \
            + self.w16(hidden) * G3[:,6:7] + self.w17(hidden) * G3[:,7:8] 
            
        hidden = self.dropout2(self.activation1(hidden))
        result = self.w20(hidden) * G3[:,0:1] + self.w21(hidden) * G3[:,1:2] \
            + self.w22(hidden) * G3[:,2:3] + self.w23(hidden) * G3[:,3:4] \
            + self.w24(hidden) * G3[:,4:5] + self.w25(hidden) * G3[:,5:6] \
            + self.w26(hidden) * G3[:,6:7] + self.w27(hidden) * G3[:,7:8] \

        # print(x[0:,-18:-1])
            
        return result
