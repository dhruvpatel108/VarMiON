# import libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb

"""
Dataloader for old stacked data format
"""
class NumpyToPytorchDataset(torch.utils.data.Dataset):
  def __init__(self, f_branch, trunk, output, device):

    self.f_branch_torch = torch.tensor(f_branch, dtype=torch.float).to(device)
    self.trunk_torch = torch.tensor(trunk, dtype=torch.float).to(device)
    self.output_torch = torch.tensor(output, dtype=torch.float).to(device).squeeze()

  def __len__(self):
    return len(self.output_torch)

  def __getitem__(self, idx):
    return self.f_branch_torch[idx], self.trunk_torch[idx], self.output_torch[idx]
  
  
def create_dataloader(f_branch__np, trunk__np, output__np, device, batch_size, if_shuffle):
  data_object =  NumpyToPytorchDataset(f_branch__np, trunk__np, output__np, device)
  ds = DataLoader(data_object, batch_size=batch_size, shuffle=if_shuffle)
  return ds


"""
helpfer function for plotting
"""
def plot_subplots(n_samples, n_val_start, f_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst, worst_idx, if_best, best_idx):
  n_rows = n_samples
  n_cols = 5
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,20))
  for ii in range(n_rows):
    if if_worst==True:
      pp = worst_idx[ii]
    elif if_best==True:
      pp = best_idx[ii]
    else:
      pp = random_sample_idx[ii]

    for jj in range(n_cols):
      # source
      if jj==0:
        im = axs[ii, jj].imshow(f_fullfield[n_val_start+pp,:,:], cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
      # bc
      #if jj==1:
      #  im = axs[ii, jj].imshow(gb_fullfield[n_val_start+pp, :, :], cmap="inferno")
      #  fig.colorbar(im, ax=axs[ii,jj])
        #im = axs[ii, jj].plot(g_fullfield[n_val_start+pp, 0, :], color="red", label="bc0")
        #im = axs[ii, jj].plot(g_fullfield[n_val_start+pp, 1, :], color="green", label="bc1")
        #im = axs[ii, jj].plot(g_fullfield[n_val_start+pp, 2, :], color="blue", label="bc2")
        #im = axs[ii, jj].plot(g_fullfield[n_val_start+pp, 3, :], color="yellow", label="bc3")
        #axs[ii, jj].legend()
      # true pressure
      if jj==2:
        im = axs[ii, jj].imshow(u_fullfield[n_val_start+pp,:,:], cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
        #im=axs[ii, jj].imshow(output_val_np[pp*NN:(pp+1)*NN].reshape([img_size, img_size]), cmap="inferno")
        #fig.colorbar(im, ax=axs[ii,jj])
      # predicted pressure
      if jj==3:
        im = axs[ii, jj].imshow(output_predictions[pp, :, :], vmin=u_fullfield[n_val_start+pp, :, :].min(), vmax=u_fullfield[n_val_start+pp, :, :].max(), cmap="inferno")
        #im=axs[ii,jj].imshow(output_pred_val_np[pp*NN:(pp+1)*NN].reshape([img_size, img_size]), 
        #                      vmin=output_val_np[pp*NN:(pp+1)*NN].min(), vmax=output_val_np[pp*NN:(pp+1)*NN].max(), cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
      # error
      if jj==4:
        error_ = u_fullfield[n_val_start+pp, :, :] - output_predictions[pp, :, :]
        im=axs[ii, jj].imshow(error_, cmap="inferno")
        fig.colorbar(im, ax=axs[ii,jj])
        axs[ii, jj].set_title(f"{100*(np.linalg.norm(error_)/np.linalg.norm(u_fullfield[n_val_start+pp, :, :])):.2f}")

  if if_worst==True:
    wandb.log({"during_training/worst_samples": fig, "epoch": t})
  elif if_best==True:
    wandb.log({"during_training/best_samples": fig, "epoch": t})
  else:
    wandb.log({"during_training/random_samples": fig, "epoch": t})




def trainer_minibatch(neural_operator, neural_operator_name, criterion, optimizer, 
                          train_dataloader, val_dataloader, training_batch_size, val_batch_size,
                          f_fullfield, u_fullfield,
                          max_epoch, log_freq, wandb_img_freq, model_path,
                          n_val, n_val_start, Ny, Nx):

    loss_np = np.zeros((int(max_epoch/log_freq), 2))
    wandb.watch(neural_operator, criterion=criterion, log_freq=wandb_img_freq*2)
    min_val = 99999
    n_worst, n_random, n_best = 10, 10, 10
    random_sample_idx = np.random.choice(n_val, n_random, replace=False)

    for t in range(max_epoch):
        for idx, (f_train, t_train, o_train) in enumerate(train_dataloader):
          print(f"Epoch/Minibatch id = {t}/{idx}")
          # prediction and loss          
          output_pred = neural_operator(f_train, t_train)
          train_loss = criterion(output_pred, o_train)          
          # Zero out gradients, perform a backward pass, and update the weights.
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
            
          
        # logging stuff
        if (t==0 or (((t+1) % log_freq) == 0)):
            loss_np[t//log_freq, 0] = t
            loss_np[t//log_freq, 1] = train_loss
            wandb.log({"train loss": train_loss, "epoch": t})
        print(f"Epoch={t} | Training loss={train_loss:.5f}")        

        # validation
        if (t==0 or (((t+1) % wandb_img_freq) == 0)):
            val_examples_loss_list = []
            output_predictions = np.zeros([n_val, Ny, Nx])
            for val_idx, (f_val, t_val, o_val) in enumerate(val_dataloader):
                # prediction and loss                            
                output_pred_val = neural_operator(f_val, t_val)
                val_loss = criterion(output_pred_val, o_val)
                output_pred_val_np = output_pred_val.cpu().detach().numpy().squeeze()   
                output_val_np = o_val.cpu().detach().numpy()        # true
                error_output_val = output_val_np - output_pred_val_np
                output_predictions[val_idx*int(val_batch_size/(Ny*Nx)):(val_idx+1)*int(val_batch_size/(Ny*Nx)), :, :] = np.reshape(output_pred_val_np, [int(val_batch_size/(Ny*Nx)), Ny, Nx])
                for ii_val in range(int(val_batch_size/(32*32))):
                  val_examples_loss_list.append(100*(np.linalg.norm(error_output_val[Ny*Nx*ii_val:(ii_val+1)*Ny*Nx])/np.linalg.norm(output_val_np[Ny*Nx*ii_val:(ii_val+1)*Ny*Nx])))
            val_example_loss = np.array(val_examples_loss_list)
            val_l2error = np.mean(np.abs(val_example_loss))
            if val_l2error < min_val:
                min_val = val_l2error
                torch.save({
                f'{neural_operator_name}_state_dict': neural_operator.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{model_path}/BestModel{t}.pth")
            wandb.log({"val error (L2)": val_l2error, "epoch": t})
            wandb.log({"val loss": val_loss, "epoch": t})

            fig0, axs0 = plt.subplots()
            axs0.hist(val_example_loss)
            axs0.set_xlabel("val error (in %)")
            wandb.log({"during_training/val_histogram": fig0, "epoch": t})

            # 10-larget val. error samples
            worst_idx = np.argsort(val_example_loss)[-n_worst:]
            # 10-smallest val. error samples
            best_idx = np.argsort(val_example_loss)[:n_best]
            # worst 10 samples from the test set
            plot_subplots(n_worst, n_val_start, f_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=True, worst_idx=worst_idx, if_best=False, best_idx=0)    
            # random 10 samples from the test set
            plot_subplots(n_random, n_val_start, f_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=False, worst_idx=0, if_best=False, best_idx=0)         
            # best 10 samples from the test set
            plot_subplots(n_best, n_val_start, f_fullfield, u_fullfield, output_predictions, t, random_sample_idx, if_worst=False, worst_idx=0, if_best=True, best_idx=best_idx)    


    return min_val
