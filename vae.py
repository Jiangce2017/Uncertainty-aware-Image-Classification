import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE_CNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim,device,im_x,im_y):
        super(VAE_CNN, self).__init__()
        self.device = device
        self.im_x = im_x
        self.im_y = im_y
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.Encoder = CNN_Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,im_x=im_x, im_y=im_y)
        self.Decoder = CNN_Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim,output_dim=output_dim,im_x=im_x, im_y=im_y)
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std).to(self.device)
        z = mu + std * epsilon
        return z
    
    def forward(self, x):
        mu, log_var = self.Encoder(x)
        z = self.reparameterization(mu, log_var) 
        x_hat = self.Decoder(z)
        return x_hat, mu, log_var    

    
class CNN_Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim,im_x,im_y):
            super(CNN_Encoder, self).__init__()
            self.input_dim = input_dim
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) ## half spatial dimension 
            self.conv3 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv4 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.conv5 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), stride=1, padding=1)
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(im_x*im_y*hidden_dim//2, hidden_dim)
            self.layer_mean = nn.Linear(hidden_dim, latent_dim)
            self.layer_variance = nn.Linear(hidden_dim, latent_dim)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.im_x = im_x
            self.im_y = im_y


        def forward(self, x):
            # print("x shape:", x.shape)
            x = x.view(-1,self.input_dim,self.im_x,self.im_y)
            h_ = self.LeakyReLU(self.conv1(x))
            h_ = self.LeakyReLU(self.conv2(h_))
            h_ = self.maxpool(h_)
            h_ = self.LeakyReLU(self.conv3(h_))
            h_ = self.LeakyReLU(self.conv4(h_))
            h_ = self.LeakyReLU(self.conv5(h_))
            h_ = self.flatten(h_)
            h_ = self.LeakyReLU(self.dense1(h_))
            mean     = self.layer_mean(h_)
            log_var  = self.layer_variance(h_)                                                                           
            return mean, log_var

class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,output_dim,im_x,im_y):
        super(CNN_Decoder, self).__init__()

        self.dense1 = nn.Linear(latent_dim, im_x*im_y*2)
        self.dense2 = nn.Linear(im_x*im_y*2,im_x*im_y*hidden_dim//2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=(3, 3), stride=1, padding=1)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.hidden_dim = hidden_dim
        self.im_x = im_x
        self.im_y = im_y

    def forward(self, x):
        h = self.LeakyReLU(self.dense1(x))
        h = self.LeakyReLU(self.dense2(h))
        h = h.view(-1,self.hidden_dim*2,self.im_x//2,self.im_y//2)
        h = self.upsample(h)
        h = self.LeakyReLU(self.conv1(h))
        x_hat = torch.sigmoid(self.conv2(h))
        return x_hat    

    