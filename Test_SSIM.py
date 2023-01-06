# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:56:32 2021

@author: 13362
"""

import cv2
import numpy as np
import math 
import os 


def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')


def psnr1(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

def psnr(target, ref):

   target_data = np.array(target, dtype=np.float64)
   ref_data = np.array(ref,dtype=np.float64)

   diff = ref_data - target_data
   diff = diff.flatten('C')
   rmse = math.sqrt(np.mean(diff ** 2.))

   eps = np.finfo(np.float64).eps
   if(rmse == 0):
       rmse = eps
   return 20*math.log10(255.0/rmse)


def C_PSNR_SSIM():
    files = os.listdir('./Clear')    
    PSNR = 0
    SSIM = 0
    for i in range(len(files)):
        img1 = cv2.imread('./Clear/' + files[i])
        img2 = cv2.imread('./MTRBNet/' + files[i][:-4] + '_LYSNet.png')
    
        ss = calculate_ssim(img1, img2)
        ps = psnr(img1, img2)
        SSIM += ss
        PSNR += ps
    
    return PSNR/100,SSIM/100

print(C_PSNR_SSIM())