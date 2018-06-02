# coding: utf-8

import os 
import numpy as np 
import torch 
import torchvision.datasets as dset  
import torch.nn as nn 
import torchvision.transforms as transforms 

import pyro 
import pyro.distributions as dist 
from pyro.infer import SVI, Trace_ELBO 
from pyro.optim import Adam

"""
pyro.enable_validation(True)
pyro.distributions.enable_validation(True)
"""
pyro.set_rng_seed(0)
smoke_test = 'CI' in os.environ 


def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True 
    trans = transforms.ToTensor() # normalize the pixel intensities 
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    kwargs = {'num_workers':1, 'pin_memory':use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

# define decoder
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used 
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img 

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used 
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale


class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim 
    
    # define the model p(x|z)p(z)
    def model(self, x):
        pyro.module("decoder", self.decoder)
        # pyro.iarange: 条件付き独立を通知
        with pyro.iarange("data", x.size(0)):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.size(0), self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.z_dim)))
            # sample from prior (value will be sampled by guide when
            #  computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images 
            pyro.sample("obs", dist.Bernoulli(loc_img).independent(1), obs=x.reshape(-1, 784))

            return loc_img
    
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register pytorch module encoder with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.iarange("data", x.size(0)):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z 
            pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
    
    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image 
        loc_img = self.decoder(z)

        return loc_img

def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0. 
    # do a training epoch over each minibatch x returned 
    # by the data loader 
    for _, (x, _) in enumerate(train_loader):
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)
    
    # return epoch loss 
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator 
    test_loss = 0. 
    # compute the loss over the entire test set 
    for i, (x, _) in enumerate(test_loader):
        if use_cuda:
            x = x.cuda()
        # compute elbo estimate and accumulate loss 
        test_loss += svi.evaluate_loss(x)
    
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


if __name__ == "__main__":
    # run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = False 

    # run only for a single iteration for testing
    NUM_EPOCHS = 1 if smoke_test else 100
    TEST_FREQUENCY = 5

    train_loader, test_loader = setup_data_loaders(
        batch_size=256, use_cuda=USE_CUDA)
    
    vae = VAE(use_cuda=USE_CUDA)
    adam_args = {"lr":LEARNING_RATE}
    optimizer = Adam(adam_args)

    # setup the inference algorithm 
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d] average training loss: %.4f" % (
            epoch, total_epoch_loss_train))
        if epoch % TEST_FREQUENCY == 0:
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (
                epoch, total_epoch_loss_test))
