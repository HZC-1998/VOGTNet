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

