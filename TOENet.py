# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F



class TOENet(nn.Module):
	def __init__(self):
		super(TOENet,self).__init__()

		self.mns = MainNetworkStructure(3,8)
         
	def forward(self,x):
        
		Fout = self.mns(x) + x
      
		return Fout


class MainNetworkStructure(nn.Module):
	def __init__(self,inchannel,channel):
		super(MainNetworkStructure,self).__init__()
        

		self.cfceb_l = CFCEB(channel)
		self.cfceb_m = CFCEB(channel*2)
		self.cfceb_s = CFCEB(channel*4)        

		self.ein = BB(channel)        
		self.el = BB(channel)
		self.em = BB(channel*2)
		self.es = BB(channel*4)
		self.ds = BB(channel*4)
		self.dm = BB(channel*2)
		self.dl = BB(channel)
        
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   

		self.conv_r_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_r_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False) 
        
		self.conv_g_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_g_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False) 
        
		self.conv_b_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_b_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False) 
        
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

		self.conv_r_in = nn.Conv2d(1,channel,kernel_size=1,stride=1,padding=0,bias=False)    
		self.conv_g_in = nn.Conv2d(1,channel,kernel_size=1,stride=1,padding=0,bias=False)    
		self.conv_b_in = nn.Conv2d(1,channel,kernel_size=1,stride=1,padding=0,bias=False)            
		self.conv_in = nn.Conv2d(inchannel,channel,kernel_size=1,stride=1,padding=0,bias=False)    
        
		self.conv_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)    
    		

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x):
        
		r = self.conv_r_in(x[:,0,:,:].unsqueeze(1))
		g = self.conv_g_in(x[:,1,:,:].unsqueeze(1))    
		b = self.conv_b_in(x[:,2,:,:].unsqueeze(1))
      
		x_r_l, x_g_l, x_b_l, x_out_l = self.cfceb_l(r,g,b)        
		x_r_m, x_g_m, x_b_m, x_out_m = self.cfceb_m(self.conv_r_eltem(self.maxpool(x_r_l)), self.conv_g_eltem(self.maxpool(x_g_l)), self.conv_b_eltem(self.maxpool(x_b_l))) 
		_, _, _, x_out_s = self.cfceb_s(self.conv_r_emtes(self.maxpool(x_r_m)), self.conv_r_emtes(self.maxpool(x_g_m)), self.conv_r_emtes(self.maxpool(x_b_m))) 
        
		x_elin = self.ein(self.conv_in(x))
		elout = self.el(x_elin * x_out_l)
		x_emin = self.conv_eltem(self.maxpool(elout))      
		emout = self.em(x_emin * x_out_m)
		x_esin = self.conv_emtes(self.maxpool(emout))          
		esout = self.es(x_esin * x_out_s)
		dsout = self.ds(esout)       
		x_dmin = self._upsample(self.conv_dstdm(dsout),emout) + emout
		dmout = self.dm(x_dmin)
		x_dlin = self._upsample(self.conv_dmtdl(dmout),elout) + elout    
		dlout = self.dl(x_dlin)
		x_out = self.conv_out(dlout)

		return x_out
    
    

class CFCEB(nn.Module):    #Channel Feature Correlation Extraction Block (CFCEB)
	def __init__(self,channel):
		super(CFCEB,self).__init__()

		self.bb_R = BB(channel)
		self.bb_G = BB(channel)
		self.bb_B = BB(channel)

		self.cab = CAB(2*channel)
		#self.cab_G = CAB(2*channel)
		#self.cab_B = CAB(2*channel)
		self.cab_RGB = CAB(3*channel)
        
		self.conv_out1 = nn.Conv2d(channel*2,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_out2 = nn.Conv2d(channel*3,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def forward(self,r,g,b):
        
		x_r = self.bb_R(r)
		x_g = self.bb_G(g)
		x_b = self.bb_B(b)

		x_r_a = self.conv_out1(self.cab(torch.cat((x_r,x_g),1))) #+ x_r + x_g
		x_g_a = self.conv_out1(self.cab(torch.cat((x_r,x_b),1))) #+ x_r + x_b
		x_b_a = self.conv_out1(self.cab(torch.cat((x_g,x_b),1))) #+ x_g + x_b 
		x_rgb_a = self.cab_RGB(torch.cat((x_r,x_g,x_b),1))#*torch.cat((x_r,x_g,x_b),1)
        
		x_out = self.conv_out2(torch.cat((x_r_a , x_g_a , x_b_a),1)+x_rgb_a) # + x_r + x_g + x_b 

		return	x_r, x_g, x_b, x_out
    
    
class BB(nn.Module):    #Basic Block (BB)
	def __init__(self,channel,norm=False):                                
		super(BB,self).__init__()

		self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1(x)))
		x_2 = self.act(self.norm(self.conv_2(x_1)))
		#x_3 = self.act(self.norm(self.conv_3(x_2)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)

		return	x_out
    
      
class CAB(nn.Module):    #Channel Attention Block (CAB)
    def __init__(self , in_planes , ration = 4):
        super(CAB, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 通道数不变，H*W变为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1) #

        self.fc1 = nn.Conv2d(in_planes , in_planes // ration , 1 , bias = False)
        self.act1 = nn.PReLU(in_planes // ration)
        self.fc2 = nn.Conv2d(in_planes // ration , in_planes , 1 , bias = False)
        self.act2 = nn.PReLU(in_planes)
        self.sigmoid = nn.Sigmoid()
        self.norm1 = nn.GroupNorm(num_channels=in_planes // ration,num_groups=1)# nn.InstanceNorm2d(channel)#
        self.norm2 = nn.GroupNorm(num_channels=in_planes,num_groups=1)# nn.InstanceNorm2d(channel)#
        self.fout = nn.Conv2d(in_planes , in_planes//2, 1 , bias = False)
        
    def forward(self , x):
        avg_out = self.norm2(self.fc2(self.act1(self.norm1(self.fc1(self.avg_pool(x))))))
        max_out = self.norm2(self.fc2(self.act1(self.norm1(self.fc1(self.max_pool(x))))))
        camap = self.sigmoid(avg_out + max_out)# * x + x   #camap: channel attention map
        #out = self.fout(camap)
        
        return camap

        