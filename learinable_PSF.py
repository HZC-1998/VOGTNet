import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
def get_kernel(factor=4, kernel_type='gauss', phase=0, kernel_width=32, support=None, sigma=3):

    # factor  = float(factor)

    kernel = np.zeros([kernel_width, kernel_width])


    center = (kernel_width + 1.) / 2.
    # print(center, kernel_width)
    sigma_sq = sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center) / 2.
            dj = (j - center) / 2.
            kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
            kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    kernel /= kernel.sum()
    return kernel


KS = 4
factor = 4
kernel = torch.rand(4, 4, KS, KS)
kernel[0, 0, :, :] = torch.from_numpy(get_kernel(factor, 'gauss', 0, KS, sigma=3))
Conv = nn.Conv2d(4, 4, KS, factor)
Conv.weight = nn.Parameter(kernel)
dow =Conv
#down_spa = Apply(dow, 1)
inf_img=torch.torch.rand(8,4,32,32)
output=dow(inf_img)
print(output.shape)


