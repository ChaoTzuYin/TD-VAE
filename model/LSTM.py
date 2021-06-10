"""
Created on Thu Jun 10 14:56:50 2021

@author: Chao, Tzu-Yin
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, cell=nn.LSTMCell, sequence_channel=1):
        '''
        input_size : The number of expected features in the input
        hidden_size : The number of features in the hidden state h
        num_layers : Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        bias : If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        cell : nn.Module, your reccurent body
        sequence_last : Sequence length should be in the corresponding dimension of x
        '''
        super().__init__()
        
        module_list = []
        in_ = input_size
        out_ = hidden_size
        
        for _ in range(num_layers):
            module_list += [cell(in_, out_)]
            in_ = out_
        
        self.net = nn.ModuleList(module_list)
        self.sequence_channel = sequence_channel
    
    def step(self, x, state=None):
        # x: [Batch, Channel]
        # state: either None or tuple of tensors (h, c) with shape [LayerNum, Batch, hidden]
        
        if(state is None):
            state = [None for _ in range(len(self.net))]
        else:
            Hs, Cs = torch.unbind(state[0],0), torch.unbind(state[1],0)
            state = [(h_, c_) for h_, c_ in zip(Hs, Cs)]
        
        h_out = []
        c_out = []
        
        #for every layers of stack LSTM
        
        h = x
        
        for fn_, s_ in zip(self.net, state):
            h, c = fn_(h, s_)
            h_out += [h]
            c_out += [c]
        
        step_new_state = (torch.stack(h_out,0),
                          torch.stack(c_out,0))
        step_out = h_out[-1]
        
        return step_out, step_new_state
    
    def forward(self, x, state=None, return_full_states=False):
        # x: [Batch, SequenceLen, Channel]
        # state: either None or tuple of tensors (h, c) with shape [LayerNum, Batch, hidden]
        
        x_1_to_T = torch.unbind(x, self.sequence_channel)
        
        out_t = []
        state_t = []
        
        for xt in x_1_to_T:
            out, state = self.step(xt, state)
            out_t += [out]
            state_t += [state]
        
        ret_out = torch.stack(out_t, self.sequence_channel)
        
        if(return_full_states == True):
            ret_state = (torch.stack([S[0] for S in state_t], self.sequence_channel+1), 
                         torch.stack([S[1] for S in state_t], self.sequence_channel+1))
        else:
            ret_state = state_t[-1]
            
        return ret_out, ret_state
    
    
    
'''
#property check
#1.The module should be able to work with second-order gradient

a = torch.autograd.Variable(torch.ones([1,3,37]).cuda(), requires_grad=True)
f = LSTM(37,64,4).cuda()
lstm_out, lstm_state = f(a)
grads_val = torch.autograd.grad(outputs=lstm_out, inputs=a,
                                grad_outputs=torch.ones_like(lstm_out),
                                create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

torch.mean(grads_val).backward()

#2.The output should be the same as that of nn.LSTM

f_torch = nn.LSTM(37, 64, 4, batch_first=True).cuda()
for param_cur, param_best in zip(f.parameters(), f_torch.parameters()):
    param_cur.data = torch.ones_like(param_cur.data)
    param_best.data = torch.ones_like(param_best.data)

    
lstm_out_torch, lstm_state_torch = f_torch(a)
lstm_out, lstm_state = f(a)
assert torch.sum((lstm_out_torch!=lstm_out).float())==0
assert torch.sum((lstm_state_torch[0]!=lstm_state[0]).float())==0
assert torch.sum((lstm_state_torch[1]!=lstm_state[1]).float())==0
'''