# Import libraries
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchinfo import summary
from models_varmion import *
from torch_rbf import *
from utils_varmion import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 1008)
np.random.seed(hash("improves reproducibility") % 1008)
torch.manual_seed(hash("by removing stochasticity") % 1008)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 1008)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Loading data
sparse_data_dir = "sparse_data/sensorgridstructured_Nf1024_Nout224/"
fullfield_data_dir = "fullfield_data/"

f_branch_data = np.load(f"{sparse_data_dir}/f_branch_data.npy")
trunk_data = np.load(f"{sparse_data_dir}/trunk_data.npy")
output_data = np.load(f"{sparse_data_dir}/output_data.npy")
f_idx = np.load(f"{sparse_data_dir}/f_idx.npy")
f_fullfield = np.load(f"{fullfield_data_dir}/f_fullfield.npy")
u_fullfield = np.load(f"{fullfield_data_dir}/u_fullfield.npy")

# =================================================
Ny = Nx = 32
n_output_sensor = 224     # no. of temp. sensors in training data
n_f_sensors = 1024        # no. of f sensors
n_train = 8000            # no. of training samples
n_val = 2000              # no. of validation samples
n_val_start = 8000        # no. of samples after which in the combined dataset the validation set starts (by default it should be n_train, but we are specifically 
                          # defining it here to consider the cases with lean training i.e. smaller training and/or validation sets)
# ==================== train ======================
f_branch_train_np = f_branch_data[:n_output_sensor*n_train, :]
trunk_train_np = trunk_data[:n_output_sensor*n_train, :]
output_train_np = output_data[:n_output_sensor*n_train, :]

# ================== validation ===================
Nx = Ny = 32
# sensor node values
f_at_sensor = f_fullfield[n_val_start:n_val_start+n_val, f_idx[:, 0], f_idx[:, 1]]
f_branch_val_np = np.repeat(f_at_sensor, repeats=Ny*Nx, axis=0)

xvec_domain, yvec_domain = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
trunk_val_np = np.tile(np.array([xvec_domain.flatten(), yvec_domain.flatten()]).T, (n_val, 1))
output_val_2d = u_fullfield[n_val_start:n_val_start+n_val, :, :].reshape([-1, 32*32])
output_val_np = np.expand_dims(output_val_2d.flatten(), axis=1)


# Print out basic information about different arrays
def log_info(array, array_name):
  print(f"{array_name}: shape={array.shape}, max={array.max()}, min={array.min()}")

log_info(f_branch_train_np, "f_branch_train_np")
log_info(trunk_train_np, "trunk_train_np")
log_info(output_train_np, "output_train_np")
print("==============")
log_info(f_branch_val_np, "f_branch_val_np")
log_info(trunk_val_np, "trunk_val_np")
log_info(output_val_np, "output_val_np")
print("==============")
log_info(f_branch_data, "f_branch_data")
log_info(trunk_data, "trunk_data")
log_info(output_data, "output_data")
print("==============")


# wandb
import wandb
wandb.login(key="") 

  
"""
wandb hyperparameter dictionary
"""
sweep_config = {"method": "grid"}
metric = {"name": "valset_l2_error",
          "goal": "minimize"}
sweep_config["metric"] = metric

max_epoch = 1000              # use at least 500 epochs
parameters_dict = {
    "max_epoch": {"values": [max_epoch]},
    "log_freq": {"values": [int(0.1*max_epoch)]},
    "wandb_img_freq": {"values": [int(0.1*max_epoch)]},
    "training_batch_size": {"values": [n_output_sensor*int(n_train/20)]},
    "val_batch_size": {"values": [1024*int(n_val/10)]},
    "latent_dim": {"values": [100]},
    "branch_h_dim": {"values": [130]}
    }
sweep_config["parameters"] = parameters_dict

import pprint
pprint.pprint(sweep_config)
project_name = "Eikonal_F_to_U_Nf1024_Aug23"     # name of the wandb project
group_name = "deeponet"       # group name of the experiment for wandb
model_dir = f"./models/{group_name}"             # directory name of the Google Drive folder
import os
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
sweep_id = wandb.sweep(sweep_config, project=f"{project_name}")



# ==================== train ======================

import time
t1 = time.time()
# helper function
def count_parameters(model):
    if type(model)==list:
      #print("The provided object is a list")
      count = 0
      for i in range(len(model)):
        count += sum(p.numel() for p in model[i] if p.requires_grad)
    return count



def run_trainer(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, group=f"{group_name}", 
                    tags=["DeepONet", f"train{n_train}_test{n_val}", "fixedLR", "MiniBatchTraining", "RBFTrunk"]):
        config = wandb.config

        # hyper-parameters for network architecture and training
        max_epoch = config.max_epoch
        log_freq = config.log_freq
        wandb_img_freq = config.wandb_img_freq
        training_batch_size, val_batch_size = config.training_batch_size, config.val_batch_size
        print(f"training_batch_size: {training_batch_size}, val_batch_size: {val_batch_size}")
        # model path to the saved model
        model_path = f"{model_dir}/{wandb.run.name}"
        if not os.path.exists(model_path):
          os.makedirs(model_path)

        # branch and trunk Networks
        branchnet_varmion = FullyConnectedNN([n_f_sensors, config.branch_h_dim, config.latent_dim]).to(device)
        #trunknet_varmion = FullyConnectedNN([2, config.latent_dim, config.latent_dim, config.latent_dim, config.latent_dim, config.latent_dim]).to(device)
        trunknet_varmion = RBF(2, config.latent_dim, gaussian).to(device)
        neural_operator = VarMiON(branchnet_varmion, trunknet_varmion).to(device)
        neural_operator_name = "deeponet"

        # optimization 
        criterion = torch.nn.MSELoss(reduction='mean')
        params_to_optimize = list(branchnet_varmion.parameters()) + list(trunknet_varmion.parameters())
        #params_to_optimize = list(mionet.parameters())
        print(f"No. of trainable params = {count_parameters(params_to_optimize)}")
        optimizer = torch.optim.Adam(params_to_optimize) 
        decayRate = 0.99
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        # data loaders
        train_dataloader = create_dataloader(f_branch_train_np, trunk_train_np, output_train_np, device, training_batch_size, if_shuffle=True)
        val_dataloader   = create_dataloader(f_branch_val_np, trunk_val_np, output_val_np, device, val_batch_size, if_shuffle=False)

        #for val_idx, (f_val, g_val, t_val, u_val, gb_val) in enumerate(val_dataloader):
        #  if val_idx == 0:
        #    print(f"val_idx = {val_idx}, f_val.shape = {f_val.shape}, g_val.shape = {g_val.shape}, t_val.shape = {t_val.shape}, u_val.shape = {u_val.shape}, gb_val.shape = {gb_val.shape}")
        
        # mini batch training and logging
        min_val = trainer_minibatch(neural_operator, neural_operator_name, criterion, optimizer, 
                          train_dataloader, val_dataloader, training_batch_size, val_batch_size,
                          f_fullfield, u_fullfield,
                          max_epoch, log_freq, wandb_img_freq, model_path,
                          n_val, n_val_start, Ny, Nx)

        # save last checkpoint
        torch.save({
                f'{neural_operator_name}_state_dict': neural_operator.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                },  f"{model_path}/End.pth")


        print(f"total elapsed time = {time.time() - t1}")
        l2_error_s = min_val   
        print(f"Min valset error in l2 sense (in %) = {min_val}")
        wandb.run.summary["valset_l2_error"] = l2_error_s
        wandb.run.summary["no_of_params"] = count_parameters(params_to_optimize)
        wandb.run.summary["no_of_branch_params"] = count_parameters(list(branchnet_varmion.parameters()))
        wandb.run.summary["no_of_trunk_params"] = count_parameters(list(trunknet_varmion.parameters()))        
"""
This cell starts training
"""
wandb.agent(sweep_id, run_trainer)