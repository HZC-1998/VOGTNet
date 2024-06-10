import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        # print(center, kernel_width)
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)

                kernel[i - 1][j - 1] = val


    else:
        assert False, 'wrong method name'

    kernel /= kernel.sum()

    return kernel
class Apply(nn.Module):
     def __init__(self, what, dim, *args):
         super(Apply, self).__init__()
         self.dim = dim

         self.what = what

     def forward(self, input):
         inputs = []
         for i in range(input.size(self.dim)):
             inputs.append(self.what(input.narrow(self.dim, i, 1)))

         return torch.cat(inputs, dim=self.dim)
     def __len__(self):
         return len(self._modules)
class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        self.kernel_size = 5
        self.kernel = torch.rand(1, 1, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x4 = x[:, 3, :, :]
        x5 = x[:, 4, :, :]
        x6 = x[:, 5, :, :]
        x7 = x[:, 6, :, :]
        x8 = x[:, 7, :, :]
        pad_size = self.kernel.size()[3]//2
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x4 = F.conv2d(x4.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x5 = F.conv2d(x5.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x6 = F.conv2d(x6.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x7 = F.conv2d(x7.unsqueeze(1), self.weight, padding=pad_size, stride=1)
        x8 = F.conv2d(x8.unsqueeze(1), self.weight, padding=pad_size, stride=1)

        x = torch.cat([x1, x2, x3, x4,x5,x6,x7,x8], dim=1)
        x = self.relu(x)

        return x
#KS = 4
#factor = 4
#kernel = torch.rand(1, 1, KS, KS)
#kernel[0, 0, :, :] = torch.from_numpy(get_kernel(factor, 'gauss', 0, KS, sigma=3))
#Conv = nn.Conv2d(1, 1, KS, factor)
#Conv.weight = nn.Parameter(kernel)
#dow = nn.Sequential(nn.ReplicationPad2d(int((KS - factor) / 2.)), Conv)
#downs = Apply(dow, 1)
#vis_img=torch.rand(8,8,32,32)
#model=GaussianBlur()
#out=model(vis_img)
#print(out.size())