import torch.nn as nn
class ARM(nn.Module):

    def __init__(self):
        super(ARM, self).__init__()
        self.Conv1 = nn.Conv2d(4, 4, 3, 1, 1)
        self.Conv2 = nn.Conv2d(4, 4, 3, 1, 1)
        self.Conv3 = nn.Conv2d(4, 4, 3, 1, 1)
        self.Conv4 = nn.Conv2d(4, 4, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))
        out = M*out + x

        return out