import torch
from torchvision import datasets

class Moving_MNIST():
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(root='/home/user/tzuyin/data/', download=False, train=train)
        self.x = self.dataset.data / 255.
        
        patches = torch.chunk(self.x,7,-1)
        moving_sequence = []
        
        for i in range(7):
            moving_sequence += [torch.cat([patches[int(count%7)] for count in range(i,i+7)],-1)]
        self.x_seq = torch.stack(moving_sequence,-1)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        random_start = torch.randint(0,7,[1])
        shuffle_idx = (torch.range(0,6) + random_start)%7
        current_data = self.x_seq[idx]
        reshape = current_data.reshape([current_data.shape[0]*current_data.shape[1],
                                        current_data.shape[2]])
        reshape_shuffle = reshape[...,shuffle_idx.long()]
        
        return reshape_shuffle, reshape_shuffle[...,1:]




