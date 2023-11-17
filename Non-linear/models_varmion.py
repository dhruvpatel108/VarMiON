# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Define a fully connected neural network which takes as an input a list of number of neurons in each layer
class FullyConnectedNN(nn.Module):
    def __init__(self, n_neurons):
        super(FullyConnectedNN, self).__init__()
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers-1):
            self.layers.append(nn.Linear(self.n_neurons[i], self.n_neurons[i+1]))

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.layers[i](x)
            if i < self.n_layers-2:
                x = F.relu(x)
        return x
    
#dnn = FullyConnectedNN([256+64, 128, 128, 128, 64])
#summary(dnn, input_size=(1, 256+64), device='cpu')


# DeepONet model
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_in, trunk_in):
        branch_out = self.branch_net(branch_in)
        trunk_out = self.trunk_net(trunk_in)
        return torch.sum((branch_out*trunk_out), axis=1)
    

"""
# MiONet model
class MiONet(nn.Module):
    def __init__(self, f_branch_net, g_branch_net, trunk_net):
        super(MiONet, self).__init__()
        self.f_branch_net = f_branch_net
        self.g_branch_net = g_branch_net
        self.trunk_net = trunk_net

    def forward(self, f_branch_in, g_branch_in, trunk_in):
        f_branch_out = self.f_branch_net(f_branch_in)
        g_branch_out = self.g_branch_net(g_branch_in)
        trunk_out = self.trunk_net(trunk_in)
        return torch.sum((f_branch_out*g_branch_out*trunk_out), axis=1)
""" 


# Variationally Mimetic Operator Network (VarMiON) model
class VarMiON(nn.Module):
    def __init__(self, branch_net_varmion, trunk_net_varmion):
        super(VarMiON, self).__init__()
        self.branch_net_varmion = branch_net_varmion
        self.trunk_net_varmion = trunk_net_varmion

    def forward(self, f_branch_in, trunk_in):
        branch_out = self.branch_net_varmion(f_branch_in)
        trunk_out = self.trunk_net_varmion(trunk_in)
        return torch.sum((branch_out*trunk_out), axis=1)







"""
class BranchNet_VarMiON(nn.Module):
    def __init__(self, n_f_sensors, r, p):
        super(BranchNet_VarMiON, self).__init__()
        self.n_f_sensors = n_f_sensors
        self.r = r
        self.p = p     
        self.f_layer = nn.Linear(self.n_f_sensors, self.p, bias=False)   
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.p, self.r))
        self.layers.append(nn.Linear(self.r, self.r))
        self.layers.append(nn.Linear(self.r, self.r))
        self.layers.append(nn.Linear(self.r, self.p))
        self.layers.append(nn.Linear(self.p, self.p))
        self.layers.append(nn.Linear(self.p, self.p))

    def forward(self, f_branch_in, g_branch_in):
        f_branch_out = self.f_layer(f_branch_in)
        x = torch.cat((f_branch_out, g_branch_in), axis=1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers)-1:
                x = F.relu(x)
        return x




class TrunkNet_VarMiON(nn.Module):
    def __init__(self, n_layers, d_in, d_hidden, d_out):
        super(TrunkNet_VarMiON, self).__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.d_in, self.d_hidden))
        for i in range(self.n_layers-2):
            self.layers.append(nn.Linear(self.d_hidden, self.d_hidden))
        self.layers.append(nn.Linear(self.d_hidden, self.d_out))

    def forward(self, trunk_in):
        x = trunk_in
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers-1:
                x = F.relu(x)
        
        # Enforce the Dirichlet boundary condition
        bnd_fn = (trunk_in[:, 0]*trunk_in[:, 1]*(1-trunk_in[:, 0])*(1-trunk_in[:, 1])).unsqueeze(dim=1)
        x = x*bnd_fn
        return x


from torchinfo import summary

branch_net = FullyConnectedNN([1024, 130, 100])
trunk_net = FullyConnectedNN([2, 100, 100, 100, 100, 100])
varmion = VarMiON(branch_net, trunk_net)
print("$$$$ BranchNet $$$$")
summary(branch_net, input_size=(1, 1024), device='cpu')
print("$$$$ TrunkNet $$$$")
summary(trunk_net, input_size=(1, 2), device='cpu')
print("$$$$ VarMiON $$$$")
summary(varmion, input_size=[(1, 1024), (1, 2)], device='cpu')
"""