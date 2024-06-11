import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.interpolate import interp2d
import glob
import os
import scipy.io as sio
import torch

def modle1(mode=2):
    if mode == 1:
        B_iso = {
            "mode": "blur",
            "kernel_size": random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
            "is_aniso": False,
            "sigma": random.uniform(0.1, 2.8),
        }
    elif mode == 2:
        B_iso = {
            "mode": "blur",
            "kernel_size": random.choice([7,9,11,13,15,17,19,21]),
            "is_aniso": True,
            "x_sigma": random.uniform(0.1, 0.5),
            "y_sigma": random.uniform(0.1, 0.5),
            "rotation": random.uniform(0, 180)
        }
    return  B_iso

def get_blur(img, degrade_dict):
    k_size = degrade_dict["kernel_size"]
    if degrade_dict["is_aniso"]:
        sigma_x = degrade_dict["x_sigma"]
        sigma_y = degrade_dict["y_sigma"]
        angle = degrade_dict["rotation"]
    else:
        sigma_x = degrade_dict["sigma"]
        sigma_y = degrade_dict["sigma"]
        angle = 0

    kernel = np.zeros((k_size, k_size))
    d = k_size // 2
    for x in range(-d, d+1):
        for y in range(-d, d+1):
            kernel[x+d][y+d] = get_kernel_pixel(x, y, sigma_x, sigma_y)
    M = cv2.getRotationMatrix2D((k_size//2, k_size//2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
    kernel = kernel / np.sum(kernel)

    # kernel = kernel*255/np.max(kernel)
    # kernel = kernel.astype(np.uint8).reshape((k_size, k_size, 1))
    # cv2.imwrite("test.png", kernel)
    img = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='reflect')

    return img

def get_kernel_pixel(x, y, sigma_x, sigma_y):
    return 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-((x*x/(2*sigma_x*sigma_x))+(y*y/(2*sigma_y*sigma_y))))


def BCIMS(filename,n_factor,pave_save,x):
    imgms = sio.loadmat(filename)['I_MS']
    imgms=imgms.astype(np.float32)
    test_blur = modle1(2)

    imgms_down = cv2.resize(imgms, (int(imgms.shape[1] / n_factor), int(imgms.shape[0] / n_factor)),interpolation=cv2.INTER_LINEAR)
    noise1 = np.random.normal(0, x/1000,imgms_down.shape)
    imgms_down=imgms_down+noise1

    sio.savemat(pave_save , {'imgMS': imgms_down})

def BCIPAN(filename,n_factor,pave_save,x):

    imgpan=sio.loadmat(filename)['I_PAN']
    imgpan = imgpan.astype(np.float32)
    imgpan=imgpan[:, :, np.newaxis]
    test_blur = modle1(2)
    imgpan = get_blur(imgpan, test_blur)
    imgpan_down=cv2.resize(imgpan,(256, 256),interpolation = cv2.INTER_AREA)
    noise1 = np.random.normal(0, x / 1000, imgpan_down.shape)
    imgpan_down = imgpan_down + noise1
    sio.savemat(pave_save , {'imgPAN':  imgpan_down})



if __name__ == "__main__":

 save_pathMS=r''
 save_pathPAN=r''
 MS_path=r''
 PAN_path=r''
 BCIMS(MS_path, 4,  save_pathMS)
 BCIPAN(PAN_path,4,save_pathPAN)


