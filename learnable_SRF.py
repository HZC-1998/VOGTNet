import numpy as np
import os
import xlrd
import torch
import torch.nn as nn
def get_spectral_response(xls_path):

    if not os.path.exists(xls_path):
        raise Exception("spectral response path does not exist")
    data = xlrd.open_workbook(xls_path)
    table = data.sheets()[0]
    num_cols = table.ncols
    cols_list = [np.array(table.col_values(i)).reshape(-1, 1) for i in range(0, num_cols)]
    sp_data = np.concatenate(cols_list, axis=1)
    sp_data = sp_data / (sp_data.sum(axis=0))  # normalize the sepctral response
    return sp_data

def get_sp_range(sp_matrix):
    HSI_bands, MSI_bands = sp_matrix.shape

    assert(HSI_bands>MSI_bands)
    sp_range = np.zeros([MSI_bands,2])
    for i in range(0,MSI_bands):
        index_dim_0, index_dim_1 = np.where(sp_matrix[:,i].reshape(-1,1)>0)
        sp_range[i,0] = index_dim_0[0]
        sp_range[i,1] = index_dim_0[-1]
    return sp_range
class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channel, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()
        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:,1] - self.sp_range[:,0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        #print(self.length_of_each_band)
        # import ipdb
        # ipdb.set_trace()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x,1,1,1,0,bias=False) for x in self.length_of_each_band])
        # self.scale_factor_net = nn.Conv2d(1,1,1,1,0,bias=False)
    def forward(self, input):
        # batch,channel,height,weight = list(input.size())
        # scaled_intput = torch.cat([self.scale_factor_net(input[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
           # print(i)
            input_slice = scaled_intput[:,self.sp_range[i,0]:self.sp_range[i,1]+1,:,:]
            #print(input_slice.size())
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list,1).clamp_(0,1)
#xls_path = 'C://Users//A//Downloads//UDALN_GRSL-master//UDALN_GRSL-master//data//spectral_response//houston18.xls'
#sp_matrix=get_spectral_response(xls_path)
#vis_img=torch.rand(1,8,32,32)
#sp_range=get_sp_range(sp_matrix)
#sp_range=[[ 0, 3]]
#down=convolution_hr2msi(4,1,sp_range)
#a=down(vis_img)
#print(a.shape)
