##############################################################
### Examples of Machine Learning Model used in demo.ipynb  ###
##############################################################

# Author: Linus Groner, 2018


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10WithReLU(nn.Module):#CIFAR-C with Softplus replced by ReLu
    def __init__(self):
        super(CIFAR10WithReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10WithSoftPlus(nn.Module):#CIFAR-C
    def __init__(self):
        super(CIFAR10WithSoftPlus, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = self.pool(self.sp(self.conv1(x)))
        x = self.pool(self.sp(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.sp(self.fc1(x))
        x = self.sp(self.fc2(x))
        x = self.fc3(x)
        return x


class OneHiddenLayer(nn.Module):#
    def __init__(self, nonlinearity_hidden, nonlinearity_out,d_in,d_hidden,d_out):
        super(OneHiddenLayer, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(d_in,d_hidden),
                                            nonlinearity_hidden,
                                            torch.nn.Linear(d_hidden,d_out),
                                            nonlinearity_out,)
    
    def forward(self,x):
        encoded = self.encoder(x)
        return encoded

class MNISTAutoencoder(nn.Module):#MNIST-A
    
    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28*28,512,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(512,256,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(256,128,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(128,32,bias=True),
                                      torch.nn.Softplus(beta=1),)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32,128,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(128,256,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(256,512,bias=True),
                                      torch.nn.Softplus(beta=1),
                                      torch.nn.Linear(512,28*28,bias=True),
                                      torch.nn.Sigmoid(),)
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MNISTAutoencoderWithReLU(nn.Module):#MNIST-A with Softplus replced by ReLu
    def __init__(self):
        super(MNISTAutoencoderWithReLU, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28*28,512,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(512,256,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(256,128,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128,32,bias=True),
                                      torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(torch.nn.Linear(32,128,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128,256,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(256,512,bias=True),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(512,28*28,bias=True),
                                      torch.nn.ReLU())
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
