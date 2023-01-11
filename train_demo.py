# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""

import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import scipy.misc
from LYSNet import *
from makedataset import Dataset
import utils_train
from Test_SSIM import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir):
	if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		print('loading existing model ......', checkpoint_dir + 'checkpoint.pth.tar')
		net = LYSNet()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']
		
	else:
		net = LYSNet()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		cur_epoch = 0
		
	return model, optimizer,cur_epoch


def save_checkpoint(state, is_best, PSNR,SSIM,filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'PSNR_%.4f_SSIM_%.4f_'%(PSNR,SSIM) + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')


        
def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr

def SyntheticHaze(img,depth):
    
    A = np.random.uniform(0.8,1.0)       
    beta = np.random.uniform(0.1,0.6)
    
    T = np.exp(-beta*depth)
    hazyimg = np.zeros((img.shape))

    for i in range(3):  
        hazyimg[i,:,:] = img[i,:,:]*T + A*(1-T)
        
    return hazyimg


def SyntheticSand(img,depth):
    
    As = [  
            [0.38824,0.58039,0.78431],
            [0.23922,0.52157,0.77647],
            [0.21569,0.44314,0.75686],
            [0.52941,0.61176,0.73333],
            [0.61176,0.66275,0.72549],
            [0.33333,0.4549,0.72549],
            [0.33725,0.55686,0.71765],
            [0.29412,0.43529,0.7098],
            [0.26275,0.47843,0.70196],
            [0.38824,0.56863,0.70196],
            [0.20784,0.44314,0.6549],
            [0.38039,0.53725,0.64706],
            [0.06275,0.2902,0.63137],
            [0.22353,0.38824,0.59608],
            [0.21961,0.47843,0.59216],
            [0.21961,0.43137,0.54902],
            [0.29804,0.40392,0.51765],
            [0.20392,0.41176,0.5098],
            [0.2902,0.43137,0.4902],
            [0.41569,0.46275,0.48627],
            [0.2,0.33725,0.43529]
        ]
    
    A = As[np.random.randint(0,len(As))]       
    beta = np.random.uniform(0.1,0.6)
    
    T = np.exp(-beta*depth)
    sandimg = np.zeros((img.shape))

    for i in range(3):  
        sandimg[i,:,:] = (img[i,:,:]+ A[i]-1)*T + A[i]
    
    return sandimg


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './dataset/Test'
	result_dir = './result'
	testfiles = os.listdir(test_dir)

    
	maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    
	print('> Loading dataset ...')
	dataset = Dataset(trainrgb=True, trainsyn=True, shuffle=True)
	loader_dataset = DataLoader(dataset=dataset, num_workers=0, batch_size=16, shuffle=True)
	count = len(loader_dataset)
	
	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir)

	L2_loss = torch.nn.MSELoss(reduce=True, size_average=True).cuda()
	color_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
	
	for epoch in range(cur_epoch,100):
		optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
		learnrate = optimizer.param_groups[-1]['lr']
		model.train()

        
		aaa = 0
		for i,data in enumerate(loader_dataset,0):

			img_c = torch.zeros(data[:,0:3,:,:].size())		
			img_l = torch.zeros(data[:,0:3,:,:].size())
			img_d = torch.zeros(data[:,0:1,:,:].size())
            
			for nx in range(data.shape[0]):             
				img_c[nx,:,:,:] = data[nx,0:3,:,:]#.numpy()
				img_d[nx,:,:,:] = data[nx,3:4,:,:]#.numpy()
          
			for nxx in range(data.shape[0]):
                
				sor = np.random.uniform(0,1)                    
				if sor <= 0.5:
					img_l[nxx] = torch.from_numpy(SyntheticHaze(img_c[nxx,:,:,:],img_d[nxx,:,:,:]))
				else:
					img_l[nxx] = torch.from_numpy(SyntheticSand(img_c[nxx,:,:,:],img_d[nxx,:,:,:]))

									
			input_var = Variable(img_l.cuda(), volatile=True)        
			target_final = Variable(img_c.cuda(), volatile=True)

			eout = model(input_var)

			enloss = 0.8 * L2_loss(eout,target_final) + 0.2 * torch.mean(-1 * color_loss(eout,target_final))
			optimizer.zero_grad()
			#Doptimizer.zero_grad()
            
			enloss.backward()
            
			optimizer.step()
			#Doptimizer.step()
            
			SN1_psnr = train_psnr(target_final,eout)		           
			print("[Epoch %d][%d/%d] lr :%f loss: %.4f PSNR_train: %.4f" %(epoch+1, i+1, count, learnrate, enloss.item(), SN1_psnr))
			
		for f in range(len(testfiles)):
			model.eval()
			with torch.no_grad():
				img_ccc = cv2.imread(test_dir + '/' + testfiles[f]) / 255.0
				img_h = hwc_to_chw(img_ccc)
				input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
				s = time.time()
				e_out = model(input_var)              
				e = time.time()   
				print(input_var.shape)       
				print(e-s)    
	             

				e_out = e_out.squeeze().cpu().detach().numpy()			               
				e_out = chw_to_hwc(e_out) 
			              
				temp = np.concatenate((img_ccc,e_out), axis=1)			
				cv2.imwrite(result_dir + '/' + testfiles[f][:-4] +'_%d_9'%(epoch)+'.png',np.clip(e_out*255,0.0,255.0))
				cv2.imwrite('./MTRBNet/' + testfiles[f][:-4] +'_LYSNet.png',np.clip(e_out*255,0.0,255.0))
        
		ps,ss =  C_PSNR_SSIM()

		print('PSNR_%.4f_SSIM_%.4f'%(ps,ss))
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()}, is_best=0,PSNR=ps,SSIM=ss)
			
			

