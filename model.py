import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from modules import Controller, MemPred#, BaselineNetwork
import numpy as np

"""
initialize hyperparameters
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5
lens = 0.5
decay = 0.1
cfg_cnn = [(2, 16, 5, 1, 2),(16, 32, 3, 1, 1)]
cfg_fc = [2*32*32, 800, 11]
cfg_img = [128, 32, 16, 8]
"""
define activations of LIF
"""
class AcFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()
    
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh)<lens
        return grad_input*temp.float()
    
act_fun = AcFun.apply

def mem_update(opt, x, mem, spike, decay, recurrent=True):
    if(recurrent):
        mem = mem*decay*(1.-spike) + opt(x)
    else:
        mem = mem*decay + opt(x)
    return mem, act_fun(mem)



class FirstToSpike(nn.Module):
    """
    Parameters
    ----------
    ninp : int
        number of features in the input data
    nclasses : int
        number of classes in the input labels
    nhid : int
        number of dimensions in the snn's hidden states.
    lam : float32
        earliness weight -- emphasis on earliness
    nlayers : int
        number of layers in SNN
    """
    def __init__(self, nhid, nclasses=11, lam=0.0):
        super(FirstToSpike, self).__init__()
        
        self.nhid = nhid
        self.lam = lam
        self.nclasses = nclasses
        
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        
        self.fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc2 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.Controller = Controller(nhid+1, 1)
        self.out = MemPred()
    
    def forward(self, input, epoch=0, test=False):
        """Compute halting points and predictions"""
        if test:
            self.Controller._epsilon = 0.0
        else:
            self.Controller._epsilon = self._epsilon#depend on epoch
        
        B = input.size(0)
        T = input.size(1)
        actions = []
        halt_points = -torch.ones((B, self.nclasses))
        predictions_train = torch.zeros((B, T, self.nclasses))
        predictions_test = torch.zeros((B, self.nclasses))
        log_pi = []

        c1_mem = c1_spike = torch.zeros(B, cfg_cnn[0][1], cfg_img[1], cfg_img[1], device=device)
        c2_mem = c2_spike = torch.zeros(B, cfg_cnn[1][1], cfg_img[2], cfg_img[2], device=device)
        
        h1_mem = h1_spike = torch.zeros(B, cfg_fc[1], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(B, cfg_fc[2], device=device)
        
        c_mem = c_spike = torch.zeros(B, 1, device=device)
        
        # --- for each timestep, accumulate membrane, select a set of actions ---
        for t in range(T):
            # run Base SNN on new data at step t
            x = input[:,t,:,:,:]
            # Base : cnn
            x = F.avg_pool2d(x.float(), 4)# layer-1 ds --> 128->32
            c1_mem, c1_spike = mem_update(self.conv1, x, c1_mem, c1_spike, decay, recurrent=True)# layer-2 cnn --> 16*32*32
            x = F.avg_pool2d(c1_spike, 2)# layer-3 ds --> 32->16            
            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike, decay, recurrent=True)# layer-4 cnn --> 32*16*16
            x = F.avg_pool2d(c2_spike, 2)# layer-5 ds --> 16->8
            x = x.view(B, -1)
            # Base : fc
            h1_mem,h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike, decay, recurrent=True)
            h2_mem,h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, decay, recurrent=True)#B*V
            
            # input for controller
            h2_sumspike += h2_spike
            # input for predictor
            spike_rate = h2_sumspike / (t+1)

            # prediction
            rate = self.out(spike_rate)
            
            # compute halting probability and sample an action
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1,1).repeat(B, 1)# time size B*1
            c_in = torch.cat((h2_sumspike, time), dim=1).detach()
            a_t, p_t = self.Controller(c_in, T, t)
            log_pi.append(p_t)
            

            # if a_t == 1(stop) and halt_points == -1(hasn't stop before), save halt_point and its rate
            predictions_test = torch.where((a_t == 1) & (predictions_test == 0), rate, predictions_test)
            halt_points = torch.where((halt_points == -1)&(a_t == 1), time, halt_points)

            # save predictor over all timesteps in predictions_train
            predictions_train[:,t,:] = rate
            #if t == (T-1):
            #    predictions_train = rate

            if test and (halt_points == -1).sum() == 0:
                break
            
        # if one element in the batch has not been halting, use its final prediction
        rate = torch.where(predictions_test == 0.0, rate, predictions_test).squeeze()
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze(0)

        if test:
            
            return rate, (1+halt_points).mean()/T
        else:
            self.log_pi = torch.stack(log_pi).squeeze(1).squeeze(2).transpose(0, 1)
            
            return predictions_train.squeeze(), (1+halt_points).mean()/T, rate
    
    def computeLoss(self, rate, y):
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        B = rate.size(0)
        T = rate.size(1)

        # --- compute reward ---
        _, y_hat = torch.max(rate, dim=2)
        self.r = (2*(y_hat.float().round() == y.view(B,1).repeat(1,T).float()).float()-1).detach().unsqueeze(2)
        
        #_, y_hat = torch.max(rate, dim=1)
        #self.r = (2*(y_hat.float().round() == y.float()).float()-1).detach().unsqueeze(1)


        # If you want a discount factor, that goes here!
        # It is used in the original implementation.

        # --- compute losses ---
        self.loss_r = (-self.log_pi.tril()*self.r).sum()/self.log_pi.size(0)
        
        y_ = torch.zeros(B, self.nclasses).scatter_(1, y.view(-1, 1), 1)
        #y_ = y_.repeat(1,T).view(B,T,-1)
        self.loss_c = MSE(rate[:,-1,:], y_)
        
        """here is a problem on penalty if we have one"""
        
        loss = self.loss_r + self.loss_c #+ self.lam*(self.wait_penalty)
        # It can help to add a larger weight to self.loss_c so early training
        # focuses on classification: ... + 10*self.loss_c + ...
        
        return loss
            
        
    
