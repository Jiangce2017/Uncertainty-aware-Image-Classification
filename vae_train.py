import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from pathlib import Path
import os
import os.path as osp
import csv
import matplotlib.pyplot as plt
from utils import Logger,compute_ood_metrics

# from model import get_model, MODELS
from model import EABlock, EABlockCNN
from eaae_model import EAAE
from vae import VAE_CNN

# Global parameters
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  # Device to use: 'cpu' or 'cuda'
### Specify dataset
dataset_name = 'cifar10' #'mnist', or fasion-mnst, or cifar10, or cifar100
if dataset_name == 'fashion-mnist':
    im_x=28
    im_y=28
    input_channels = 1
    output_channels = 1
    epi_channels=32
    latent_dim=32
    modes1 = 10
    modes2 = 6
    hidden_dim=32
    x_dim = im_x*im_y*input_channels
elif dataset_name == 'cifar10':
    im_x=32
    im_y=32
    input_channels = 3
    output_channels = 3
    epi_channels=32
    latent_dim=32
    modes1 = 16
    modes2 = 9
    hidden_dim=64
    x_dim = im_x*im_y*input_channels
    



# Default hyperparameters
vae_lr = 1e-4   
uncertainty_lr = 1e-4
log_interval = 10
vae_epochs = 500
uncertainty_epochs = 500
batch_size = 64
data_root = '/ocean/projects/cis240139p/jchen39/datasets' 
results_dir = Path("/jet/home/jchen39/projects/EAAE/results3")
models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints3")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

vae_model_name = 'vaecnn'
uncertainty_model_name = 'eaCNN' #'eaCNN', 'eaae'
experiment_name = dataset_name+ "_"+ uncertainty_model_name+ "_" + str(hidden_dim) + "_" + str(latent_dim)
vae_model_path = Path(os.path.join(models_dir, experiment_name+"_vae.pth"))
vae_outputs_path = Path(os.path.join(data_root, experiment_name + '_vae_outputs.npz'))


uncertainty_model_path = Path(os.path.join(models_dir, uncertainty_model_name+".pth"))
uncertainty_train_outputs_path = Path(os.path.join(data_root, experiment_name + '_uncertainty_train_outputs.npz'))
uncertainty_test_outputs_path = Path(os.path.join(data_root, experiment_name + '_uncertainty_test_outputs.npz'))
uncertainty_valid_outputs_path = Path(os.path.join(data_root, experiment_name + '_uncertainty_valid_outputs.npz'))
out_of_distribution_results_path = os.path.join(results_dir, experiment_name+'_stat_results.txt')

## setup logger
result_logger = Logger(
    osp.join(results_dir, experiment_name + '.log'),
    ['ep', 'train_epi_loss']
)

fast_test = False  # Set to True for faster benchmarking with a smaller dataset
use_old_vae_model = False
use_old_vae_data = False
use_old_uncertainty_model = False

def load_dataset(dataset_name, data_root, batch_size, fast_test=False):
    """Load dataset and return DataLoader objects for training and testing."""
    if dataset_name == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True,
                                              transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=data_root, train=False, transform=transforms.ToTensor())
        valid_dataset = datasets.MNIST(root=data_root, train=False, transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True,
                                         transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=transforms.ToTensor())
        valid_dataset = datasets.SVHN(root=data_root, split='test', download=True,
                                      transform=transforms.ToTensor())
    if fast_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(128))  # Use a subset for faster benchmarking
        test_dataset = torch.utils.data.Subset(test_dataset, range(128))  # Use a subset for faster benchmarking
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(128))  # Use a subset for faster benchmarking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader

# 2. Define the VAE Loss Function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross-Entropy for pixel values 0-1) [2]
    # print("recon_x shape:", recon_x.shape)
    # print("x shape:", x.shape)
    BCE = F.binary_cross_entropy(recon_x.view(-1, x_dim), x.view(-1, x_dim), reduction='mean')
    # KL divergence loss (penalizes deviation from standard normal distribution) [2]
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE #+ KLD*0.01

# 3. Training Loop and Parameters
def train_vae(model, train_loader):
    # Initialize model, optimizer
    optimizer = Adam(model.parameters(), lr=vae_lr)
    model.train()
    for epoch in range(1,vae_epochs+1):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{vae_epochs}], Loss: {total_loss / (batch_idx+1):.8f}')
        if epoch == vae_epochs:
            ### save vae latent vectors mu
            all_mu = []
            all_data = []
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                mu, _ = model.Encoder(data)
                all_mu.append(mu.detach().cpu().numpy())
                all_data.append(data.detach().cpu().numpy())
            all_mu = np.concatenate(all_mu, axis=0)
            all_data = np.concatenate(all_data, axis=0)
            np.savez(vae_outputs_path, mu=all_mu, data=all_data)

    # Save the model
    torch.save(model.state_dict(), vae_model_path)
    print("Training complete and model saved.")
    return all_mu, all_data

## 4. train uncertainty model
def train_uncertainty_model(uncertainty_model, train_loader, vae_model, train_mu_mean, train_mu_std, train_mu_min):
    print(f"Uncertainty Model: {uncertainty_model_name}")
    print(f"EA Block Parameters: {sum(p.numel() for p in uncertainty_model.parameters()):,}")
    optimizer = Adam(uncertainty_model.parameters(), lr=uncertainty_lr)
    uncertainty_model.train()
    vae_model.eval()
    for epoch in range(1, uncertainty_epochs+1):
        total_epi_loss = 0
        total_vae_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(data)
            vae_loss_value = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            mu = (mu - train_mu_mean) / train_mu_std
            mu = mu - train_mu_min + 0.1
            mu = mu[:,:,None,None]
            #mu = torch.sigmoid(mu)+1
            sph_err = uncertainty_model(data, mu)
            #loss = torch.mean(-torch.exp(-sph_err))
            loss = torch.mean(sph_err)
            loss.backward()
            optimizer.step()
            total_epi_loss += loss.item()
            total_vae_loss += vae_loss_value.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Loss: {total_epi_loss / (batch_idx+1):.8f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], VAE Loss: {total_vae_loss / (batch_idx+1):.8f}')
            # Log epoch results
            result_logger.log({
                'ep': epoch,
                'train_epi_loss': loss.item(),
            })
        if epoch == uncertainty_epochs: ### save uncertainty model ouputs
            all_sph_err = []
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                recon_batch, mu, logvar = vae_model(data)
                #vae_loss_value = F.binary_cross_entropy(recon_batch, data, reduce=None)
                vae_loss_value = torch.mean((recon_batch-data)**2, dim=(1,2,3))
                print("vae_loss_value shape:", vae_loss_value.shape)
                mu = (mu - train_mu_mean) / train_mu_std
                mu = mu - train_mu_min + 0.1
                mu = mu[:,:,None,None]
                #mu = torch.sigmoid(mu)+1
                sph_err = uncertainty_model(data, mu)
                
                all_sph_err.append(torch.mean(sph_err,dim=(1,2,3)).detach().cpu().numpy()+vae_loss_value.detach().cpu().numpy())
            all_sph_err = np.concatenate(all_sph_err, axis=0)
            np.savez(uncertainty_train_outputs_path, sph_err=all_sph_err)
    # Save the model
    torch.save(uncertainty_model.state_dict(), uncertainty_model_path)
    print("Training complete and model saved.")
    return total_epi_loss, all_sph_err

def test_uncertainty_model(uncertainty_model, device, test_loader, vae_model,train_mu_mean, train_mu_std,train_mu_min,mode='test'):
    uncertainty_model.eval()
    vae_model.eval()
    total_epi_loss = 0
    total_vae_loss = 0
    all_sph_err = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = vae_model(data)
            #vae_loss_value = F.binary_cross_entropy(recon_batch, data, reduce=None)
            vae_loss_value = torch.mean((recon_batch-data)**2, dim=(1,2,3))
            mu = (mu - train_mu_mean) / train_mu_std
            mu = mu - train_mu_min + 0.1
            mu = mu[:,:,None,None]
            sph_err = uncertainty_model(data, mu)
            all_sph_err.append(torch.mean(sph_err,dim=(1,2,3)).detach().cpu().numpy()+vae_loss_value.detach().cpu().numpy())
            #loss = torch.mean(-torch.exp(-sph_err))
            loss = torch.mean(sph_err)
            total_epi_loss += loss.item()
            total_vae_loss += torch.mean(vae_loss_value).item()
    print(f'Test Loss: {total_epi_loss / len(test_loader.dataset):.8f}')
    print(f'Test VAE Loss: {total_vae_loss / len(test_loader.dataset):.8f}')
    all_sph_err = np.concatenate(all_sph_err, axis=0)
    if mode == 'test':
        np.savez(uncertainty_test_outputs_path, sph_err=all_sph_err)
    elif mode == 'valid':
        np.savez(uncertainty_valid_outputs_path, sph_err=all_sph_err)
    return total_epi_loss, all_sph_err

def evaluate_vae_reconstruction(vae_model, device, data_loader):
    ### randomly sample data from training dataset and draw reconstructions
    vae_model.eval()
    ### select the first batch
    random_batch_idx = np.random.randint(0, len(data_loader))
    #for batch_idx, (data, _) in enumerate(data_loader):
        #if batch_idx == random_batch_idx:
    it = iter(data_loader)
    data, _ = next(it)
    batch_idx = 0
    with torch.no_grad():
        data = data.to(device)
        recon_data, _, _ = vae_model(data)
        # visualize or save reconstructions as needed
        # For example, you could use matplotlib to display the original and reconstructed images
        fig, axes = plt.subplots(2,6, figsize=(12, 4))
        for i in range(6):
            axes[0, i].imshow(data[i].cpu().permute(1, 2, 0))
            #axes[0, i].axis("off")
            axes[1, i].imshow(recon_data[i].cpu().permute(1, 2, 0))
            #axes[1, i].axis("off")
        ### save the picture to results_dir/constructions
        plt.savefig(results_dir / f"reconstruction_{batch_idx}.png")
        plt.close(fig)


if __name__ == '__main__':
    ## load datasets
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size, fast_test=fast_test)
    vae_model = VAE_CNN(input_channels, output_channels, hidden_dim, latent_dim,device,im_x,im_y).to(device)
    if vae_model_path.exists() and use_old_vae_model:
        print(f"Loading trained VAE model from {vae_model_path}")
        vae_model.load_state_dict(torch.load(vae_model_path))
    else:
        print("No trained VAE model found, training a new model.")

    if vae_outputs_path.exists() and use_old_vae_data and use_old_vae_model:
        vae_data = np.load(vae_outputs_path)
        all_mu = vae_data['mu']
        all_data = vae_data['data']
    else:
        all_mu, all_data = train_vae(vae_model, train_loader)
    evaluate_vae_reconstruction(vae_model, device, train_loader)
    ### print the max and min of the latent vectors
    print(f"Max of latent vectors: {np.max(all_mu)}")
    print(f"Min of latent vectors: {np.min(all_mu)}")
    print(f"Mean of latent vectors: {np.mean(all_mu)}")
    print(f"Std of latent vectors: {np.std(all_mu)}")
    ### normalize the latent vectors
    train_mu_mean = np.mean(all_mu, axis=0).reshape(1, -1)
    train_mu_std = np.std(all_mu, axis=0).reshape(1, -1)
    ### make all_mul larger than 0.1
    all_mu = (all_mu - train_mu_mean) / train_mu_std
    train_mu_min = np.min(all_mu, axis=0).reshape(1, -1)
    all_mu = all_mu - train_mu_min + 0.1
    print(f"Max of latent vectors: {np.max(all_mu)}")
    print(f"Min of latent vectors: {np.min(all_mu)}")
    print(f"Mean of latent vectors: {np.mean(all_mu)}")
    print(f"Std of latent vectors: {np.std(all_mu)}")
    train_mu_mean = torch.tensor(train_mu_mean, dtype=torch.float32).to(device)
    train_mu_std = torch.tensor(train_mu_std, dtype=torch.float32).to(device)
    train_mu_min = torch.tensor(train_mu_min, dtype=torch.float32).to(device)

    ### create a new dataloader for uncertainty model training
    all_mu_tensor = torch.tensor(all_mu, dtype=torch.float32)
    all_mu_tensor = torch.sigmoid(all_mu_tensor)+1
    all_data_tensor = torch.tensor(all_data, dtype=torch.float32)
    vae_datasets = torch.utils.data.TensorDataset(all_data_tensor, all_mu_tensor)
    uncertainty_train_loader = DataLoader(vae_datasets, batch_size=batch_size, shuffle=True)

    ### train eaae model
    if uncertainty_model_name == 'eaFNO':
        uncertainty_model = EABlockCNN(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaCNN':
        uncertainty_model = EABlockCNN(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaae':
        uncertainty_model = EAAE(im_x, im_y, hidden_dim, latent_dim, input_channels, modes1, modes2, batch_size, device)
    ## if trained uncertainty model exists, load it
    if uncertainty_model_path.exists() and use_old_uncertainty_model:
        print(f"Loading trained uncertainty model from {uncertainty_model_path}")
        uncertainty_model.load_state_dict(torch.load(uncertainty_model_path))
    else:
        print("No trained uncertainty model found, training a new model.")
    uncertainty_model = uncertainty_model.to(device)
    train_uncertainty_loss, train_uncertainty = train_uncertainty_model(uncertainty_model, train_loader,vae_model,train_mu_mean, train_mu_std,train_mu_min)
    test_uncertainty_loss, test_uncertainty = test_uncertainty_model(uncertainty_model, device, test_loader, vae_model, train_mu_mean, train_mu_std,train_mu_min,mode='test')
    valid_uncertainty_loss, valid_uncertainty = test_uncertainty_model(uncertainty_model, device, valid_loader, vae_model,train_mu_mean, train_mu_std,train_mu_min, mode='valid')

    print("Train mean uncertainty:", np.mean(train_uncertainty))
    print("Test mean uncertainty:", np.mean(test_uncertainty))
    print("Valid mean uncertainty:", np.mean(valid_uncertainty))
    print("Train median uncertainty:", np.median(train_uncertainty))
    print("Test median uncertainty:", np.median(test_uncertainty))
    print("Valid median uncertainty:", np.median(valid_uncertainty))
    print("Train min uncertainty:", np.min(train_uncertainty))
    print("Train max uncertainty:", np.max(train_uncertainty))
    print("Test min uncertainty:", np.min(test_uncertainty))
    print("Test max uncertainty:", np.max(test_uncertainty))
    print("Valid min uncertainty:", np.min(valid_uncertainty))
    print("Valid max uncertainty:", np.max(valid_uncertainty))
    train_scores = np.exp(-train_uncertainty)
    test_scores = np.exp(-test_uncertainty)
    valid_scores = np.exp(-valid_uncertainty)
    print("train_scores shape:", train_scores.shape)
    print("test_scores shape:", test_scores.shape)
    print("valid_scores shape:", valid_scores.shape)
    metrics_train_test = compute_ood_metrics(train_scores, test_scores)
    print("OOD Detection Metrics between train and test:", metrics_train_test)
    metrics_train_valid = compute_ood_metrics(train_scores, valid_scores)
    print("OOD Detection Metrics between train and valid:", metrics_train_valid)
    metrics_test_valid = compute_ood_metrics(test_scores, valid_scores)
    print("OOD Detection Metrics between test and valid:", metrics_test_valid)
    ### save the above results to in a text file
    with open(out_of_distribution_results_path, 'w') as f:
        f.write(f"Train mean uncertainty: {np.mean(train_uncertainty)}\n")
        f.write(f"Test mean uncertainty: {np.mean(test_uncertainty)}\n")
        f.write(f"Valid mean uncertainty: {np.mean(valid_uncertainty)}\n")
        f.write(f"Train median uncertainty: {np.median(train_uncertainty)}\n")
        f.write(f"Test median uncertainty: {np.median(test_uncertainty)}\n")
        f.write(f"Valid median uncertainty: {np.median(valid_uncertainty)}\n")
        f.write(f"Train min uncertainty: {np.min(train_uncertainty)}\n")
        f.write(f"Train max uncertainty: {np.max(train_uncertainty)}\n")
        f.write(f"Test min uncertainty: {np.min(test_uncertainty)}\n")
        f.write(f"Test max uncertainty: {np.max(test_uncertainty)}\n")
        f.write(f"Valid min uncertainty: {np.min(valid_uncertainty)}\n")
        f.write(f"Valid max uncertainty: {np.max(valid_uncertainty)}\n")
        f.write(f"OOD Detection Metrics between train and test: {metrics_train_test}\n")
        f.write(f"OOD Detection Metrics between train and valid: {metrics_train_valid}\n")
        f.write(f"OOD Detection Metrics between test and valid: {metrics_test_valid}\n")
    print(f"Results saved to: {out_of_distribution_results_path}")   