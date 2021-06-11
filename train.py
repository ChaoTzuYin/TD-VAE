import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from moving_mnist import Moving_MNIST
from model.tdvae_model import tdvae
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.main = nn.Sequential(
                                    nn.Conv1d(784, 256, 1),
                                    nn.ReLU(True),
                                    nn.Conv1d(256, 128, 1),
                                    nn.ReLU(True),
                                )

    def forward(self, input):
        return self.main(input)

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.main = nn.Sequential(
                                    nn.Conv1d(16*2, 256, 1),
                                    nn.ReLU(True),
                                    nn.Conv1d(256, 784, 1),
                                    nn.Sigmoid()
                                )

    def forward(self, input):
        return self.main(input)

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()
        self.state_abstraction = tdvae(input_dim=128,
                                     belief_state_dim=64, 
                                     MLP_hidden_dim=64,
                                     distribution_dim=16,
                                     num_layer_block=2,
                                     backbone_stack_layer=1)
    
    def train_forward(self, xt):
        self.train()
        feature = self.encoder(xt)
        _, z_list, _, tdvae_loss_list = self.state_abstraction(feature)
        xt_p1_pred = self.decoder(torch.cat(z_list,1))
        return xt_p1_pred, tdvae_loss_list
    
    def test_forward(self, xt):
        self.eval()
        feature = self.encoder(xt)
        _, z_list, _, _ = self.state_abstraction(feature)
        xt_p1_pred = self.decoder(torch.cat(z_list,1))
        return xt_p1_pred    
    
    def rollout(self, xt, rollout_step=14):
        self.eval()
        feature = self.encoder(xt)
        t_z_list = self.state_abstraction.rollout(feature, rollout_step)
        recon = self.decoder(torch.cat(t_z_list,1))
        return recon
        

trainSet = Moving_MNIST(True)
testSet = Moving_MNIST(False)
trainLoader = dset.DataLoader(trainSet, batch_size=500, shuffle=True, pin_memory=True, num_workers=4)
testLoader = dset.DataLoader(testSet, batch_size=500, shuffle=False, pin_memory=True, num_workers=4)

writer = SummaryWriter('./logdir')
model = FullModel().cuda()

# Parameters
MAXEPOCH = 10000
lr = 1e-3
optimizer = optim.Adam([p for p in model.parameters()], lr=1e-3)
iteration_per_epoch = len(trainLoader)
# Train
epoch = 0
for epoch in range(epoch, MAXEPOCH):
    print('Epoch ', epoch, ' start.')
    for times, data in enumerate(trainLoader):
        xt, xt_p1 = data[0].cuda(), data[1].cuda()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foward + backward + optimize
        xt_p1_pred, tdvae_loss_list = model.train_forward(xt)
        reconstruction_loss = torch.mean(torch.sum(- xt_p1 * torch.log(torch.clamp(xt_p1_pred,1e-8,1)) - (1 - xt_p1) * torch.log(torch.clamp(1 - xt_p1_pred,1e-8,1)), 1))
        
        loss = reconstruction_loss + sum(tdvae_loss_list)
        loss.backward()
        optimizer.step()

        # Print statistics
        
        writer.add_scalar('loss/reconstruction', reconstruction_loss, epoch*iteration_per_epoch+times)
        [writer.add_scalar('loss/layer'+str(len(tdvae_loss_list)-count), item, epoch*iteration_per_epoch+times) for count, item in enumerate(tdvae_loss_list)]

    writer.add_video('train/pred', xt_p1_pred[:5].permute(0,2,1).reshape([5,-1,1,28,28]).repeat(1,1,3,1,1), epoch) #N,T,C,H,W
    writer.add_video('train/gt', xt_p1[:5].permute(0,2,1).reshape([5,-1,1,28,28]).repeat(1,1,3,1,1), epoch) #N,T,C,H,W
    
    model.eval()
    rollout_result = model.rollout(xt[:5])
    writer.add_video('train/rollout', rollout_result.permute(0,2,1).reshape([5,-1,1,28,28]).repeat(1,1,3,1,1), epoch) #N,T,C,H,W

print('Training Finished.')








