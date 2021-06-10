"""
Created on Thu Jun 10 14:56:50 2021

@author: Chao, Tzu-Yin
"""

import torch
import torch.nn as nn
from LSTM import LSTM

class GateActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv1d(dim, dim, 1),
                                  nn.Sigmoid())
        self.value = nn.Sequential(nn.Conv1d(dim, dim, 1),
                                   nn.Tanh())
    def forward(self, x):
        return self.gate(x)*self.value(x)

class simple_network(nn.Module):
    def __init__(self, in_, hidden_, out_):
        super().__init__()
        self.net = nn.Sequential(
                                 nn.Conv1d(in_,hidden_,1),
                                 nn.LeakyReLU(),
                                 GateActivation(hidden_),
                                 nn.Conv1d(hidden_, 2*out_, 1, bias=False)
                                 )
    def forward(self, x):
        mu, log_sigma = torch.chunk(self.net(x),2,1)
        return torch.distributions.Normal(mu, torch.exp(log_sigma))

class layer_block(nn.Module):
    def __init__(self, backbone, PB, PT, QS):
        super().__init__()
        self.backbone_rnn = backbone
        self.PB = PB
        self.PT = PT
        self.QS = QS

    def upward_pass(self, x, state=None):
        # LSTM
        x_out, state_out = self.backbone_rnn(x, state)
        return x_out, state_out
    
    def downward_pass(self, b, zt1_l_plus_n=[], zt2_l_plus_n=[]):
        if(self.training):
            PBt2_inp = torch.cat([b[...,1:]]+zt2_l_plus_n,1) #[B, C, T-1]
            P_PBt2 = self.PB(PBt2_inp)
            z_PBt2 = P_PBt2.rsample()
            
            QS_inp = torch.cat([b[...,1:], b[...,:-1], z_PBt2]+zt1_l_plus_n,1) #[B, C, T-1]
            PBt1_inp = torch.cat([b[...,:-1]]+zt1_l_plus_n,1) #[B, C, T-1]
            
            P_QSt1 = self.QS(QS_inp)
            z_QSt1 = P_QSt1.rsample()
            
            P_PBt1 = self.PB(PBt1_inp)
            loss_PB = torch.mean(torch.distributions.kl_divergence(P_QSt1, P_PBt1))
            
            PTt1_inp = torch.cat([z_QSt1]+zt2_l_plus_n,1) #[B, C, T-1]
            P_PTt1 = self.PT(PTt1_inp)
            
            loss_PT = torch.mean(P_PBt2.log_prob(z_PBt2) - P_PTt1.log_prob(z_PBt2))
            
            return zt1_l_plus_n+[z_QSt1], zt2_l_plus_n+[z_PBt2], loss_PB + loss_PT
        
        else:
            PBt1_inp = torch.cat([b]+zt1_l_plus_n,1)
            P_PBt1 = self.PB(PBt1_inp)
            z_PBt1 = P_PBt1.sample()
            
            PTt1_inp = torch.cat([z_PBt1]+zt2_l_plus_n,1) #[B, C, T-1]
            P_PTt1 = self.PT(PTt1_inp)
            z_PTt1 = P_PTt1.sample() #zt2 predicted by PT
            
            return zt1_l_plus_n+[z_PBt1], zt2_l_plus_n+[z_PTt1], torch.zeros(1).to(z_PTt1.device)

class tdvae(nn.Module):
    def __init__(self, 
                 input_dim,
                 belief_state_dim, 
                 MLP_hidden_dim,
                 distribution_dim,
                 num_layer_block=1,
                 backbone_stack_layer=1
                 ):
        super().__init__()
        
        ind = input_dim
        Blocks = []
        
        for layer_count in range(num_layer_block):
            
            layer_backbone = LSTM(ind, belief_state_dim, 
                                  backbone_stack_layer, 
                                  sequence_channel=-1)
            
            # input: b1, all upper zPBt1
            layer_PBnet = simple_network(belief_state_dim + distribution_dim*(num_layer_block-1-layer_count), 
                                         MLP_hidden_dim,
                                         distribution_dim)
            
            # input: zt1, all upper PTzt1
            layer_PTnet = simple_network(distribution_dim + distribution_dim*(num_layer_block-1-layer_count), 
                                         MLP_hidden_dim, 
                                         distribution_dim)
            
            # input: b1, b2, PBz2, all upper QSzt1
            layer_QSnet = simple_network(belief_state_dim*2 + distribution_dim + distribution_dim*(num_layer_block-1-layer_count), 
                                         MLP_hidden_dim, 
                                         distribution_dim)
        
            Blocks = [layer_block(layer_backbone, 
                                   layer_PBnet,
                                   layer_PTnet,
                                   layer_QSnet)] + Blocks
            
            ind = belief_state_dim
            
        self.blocks = nn.ModuleList(Blocks)
        
    def reccursive_for_sampling(self, b, pass_block, state=None):
        # correctness check.
        if(state is not None):
            assert len(state) == len(pass_block), "State for all the layer blocks are required if it's not None."
        
        # check if it's the end of reccursive.
        if(len(pass_block)==0):
            return [], [], [], []
        
        # upward pass for from the belief states.
        b_out, state_out = pass_block[-1].upward_pass(b, None if state is None else state[-1])
        
        # reccursive
        ret_zt1_l_plus_n, ret_zt2_l_plus_n, ret_state, ret_loss = self.reccursive_for_sampling(b_out, pass_block[:-1], None if state is None else state[:-1])
        
        # start to downward sampling
        zt1_l_plus_n, zt2_l_plus_n, vae_loss = pass_block[-1].downward_pass(b_out, ret_zt1_l_plus_n, ret_zt2_l_plus_n)
        
        return zt1_l_plus_n, zt2_l_plus_n, ret_state + [state_out], ret_loss + [vae_loss]
    
    def forward(self, x, state=None):
        #x: sequence with shape of [B,C,S]
        #return value: list of zPBt1, zPTt1, the final state, loss
        #NOTICE: tdvae has different performance under model.train() and model.eval().
        #Please check the paper for more detial.
        return self.reccursive_for_sampling(x, self.blocks, state)

'''
#Test reccursive#
TD = tdvae(input_dim=37,
         belief_state_dim=64, 
         MLP_hidden_dim=64,
         distribution_dim=16,
         num_layer_block=16,
         backbone_stack_layer=1)

a = torch.ones([64,37,10])
TD = TD.train()
ret = TD(a)
'''