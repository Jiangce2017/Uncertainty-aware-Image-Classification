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
from utils import Logger,compute_ood_metrics,GaussianNormalizer,RangeNormalizer

# from model import get_model, MODELS
from model import EABlockFNO, EABlockCNN,CompoundModel
from eaae_model import EAAE
from vae import VAE_CNN, Freq_FNO
from resnet import BasicBlock, Bottleneck, ResNet 

# Global parameters
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  # Device to use: 'cpu' or 'cuda'

### set torch random seed for reproducibility
torch.manual_seed(0)

### Specify dataset

dataset_name = 'cifar10' #'mnist', or fasion-mnst, or cifar10, or cifar100
if dataset_name == 'fashion-mnist':
    im_x=28
    im_y=28
    input_channels = 1
    output_channels = 1
    epi_channels=10
    latent_dim=64
    modes1 = 10
    modes2 = 6
    hidden_dim=128
    num_classes = 10
    x_dim = im_x*im_y*input_channels
elif dataset_name == 'cifar10':
    im_x=32
    im_y=32
    input_channels = 3
    output_channels = 3
    epi_channels=32
    latent_dim=64
    modes1 = 10
    modes2 = 6
    num_classes = 10
    hidden_dim= 128
    x_dim = im_x*im_y*input_channels

    vae_hidden_dim = 64
    vae_latent_dim = 32
    
# Default hyperparameters
vae_lr = 1e-4   
uncertainty_lr = 1e-4
log_interval = 10
classification_epochs = 5
vae_epochs = 200
uncertainty_epochs = 2000
batch_size = 64
data_root = '/ocean/projects/cis240139p/jchen39/datasets' 
results_dir = Path("/jet/home/jchen39/projects/EAAE/results10")
models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints10")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

fast_test = True  # Set to True for faster benchmarking with a smaller dataset
use_old_uncertainty_model = True


uncertainty_model_name = 'compound' #'eaCNN', 'eaae', 'eaFNO'

experiment_name = dataset_name+ "_"+ str(hidden_dim) + "_" + str(latent_dim)+"_"+str(modes1)+"_"+str(modes2)
if fast_test:
    experiment_name += "_fast_test"

uncertainty_model_initial_path = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name + "_best.pth"))
uncertainty_model_path = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name +".pth"))

uncertainty_train_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_train_outputs.npz'))
uncertainty_test_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_test_outputs.npz'))
uncertainty_valid_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_valid_outputs.npz'))

out_of_distribution_results_path = os.path.join(results_dir, uncertainty_model_name+ "_" + experiment_name +'_stat_results.txt')



## setup logger
result_logger = Logger(
    osp.join(results_dir, uncertainty_model_name + "_" + experiment_name + '.log'),
    ['ep', 'train_class_loss', 'train_correct', 'train_epi_loss']
)

OOD_result_logger = Logger(
    osp.join(results_dir, uncertainty_model_name + "_" + experiment_name + '_OOD.log'),
    ['ep', 'train_vs_test', 'train_vs_valid', 'test_vs_valid']
)

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
        ### randomly select a subset of the dataset for faster benchmarking
        train_dataset = torch.utils.data.Subset(train_dataset, range(1024))
        # test_dataset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), size=128, replace=False))
        # valid_dataset = torch.utils.data.Subset(valid_dataset, np.random.choice(len(valid_dataset), size=128, replace=False))
    test_dataset = torch.utils.data.Subset(test_dataset, range(128))  # Use a subset for faster benchmarking
    valid_dataset = torch.utils.data.Subset(valid_dataset, range(128))  # Use a subset for faster benchmarking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader, test_loader, valid_loader


def train_classification_compound_with_uncertainty_model(model, optimizer, data_loader, device):
    model.train()
    total_class_loss = 0
    correct = 0
    total_epi_loss = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        ## convert labels to one-hot encoding
        outputs, mid_value, sph_err = model(data)
        class_loss = F.cross_entropy(outputs[:,:,0,0], labels)
        epi_loss = torch.mean(sph_err) + torch.mean((mid_value-outputs)**2)
        loss = class_loss + epi_loss
        loss.backward()
        optimizer.step()
        ### calculate the classification accuracy for this batch
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total_class_loss += class_loss.item()
        total_epi_loss += epi_loss.item()
    total_class_loss /= (batch_idx+1)
    total_epi_loss /= (batch_idx+1)
    correct /= (len(data_loader.dataset))
    return total_class_loss, correct, total_epi_loss

    #     if epoch % log_interval == 0:
    #         print(f'Epoch [{epoch}/{uncertainty_epochs}], Class Loss: {total_class_loss / (batch_idx+1):.8f}')
    #         print(f'Epoch [{epoch}/{uncertainty_epochs}], Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
    #         print(f'Epoch [{epoch}/{uncertainty_epochs}], Class Accuracy: {correct / (len(data_loader.dataset)):.4f}')
    #         ### Log epoch results
    #         result_logger.log({
    #             'ep': epoch,
    #             'train_class_loss': total_class_loss / (batch_idx+1),
    #             'train_correct': correct / (len(data_loader.dataset)),
    #             'train_epi_loss': total_epi_loss / (batch_idx+1)
    #         })
    #         train_uncertainty = evaluate_classification_compound_with_uncertainty_model(model, train_loader, device, mode='train')
    #         test_uncertainty = evaluate_classification_compound_with_uncertainty_model(model, test_loader, device, mode='test')
    #         valid_uncertainty = evaluate_classification_compound_with_uncertainty_model(model, valid_loader, device, mode='valid')
    #         train_scores = np.exp(-train_uncertainty)
    #         test_scores = np.exp(-test_uncertainty)
    #         valid_scores = np.exp(-valid_uncertainty)
    #         metrics_train_test = compute_ood_metrics(train_scores, test_scores)
    #         metrics_train_valid = compute_ood_metrics(train_scores, valid_scores)
    #         metrics_test_valid = compute_ood_metrics(test_scores, valid_scores)
    #         OOD_result_logger.log({
    #             'ep': epoch,
    #             'train_vs_test': metrics_train_test['AUROC'],
    #             'train_vs_valid': metrics_train_valid['AUROC'],
    #             'test_vs_valid': metrics_test_valid['AUROC']
    #         })
    #         uncertainty_model_path_epoch = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name + f"_epoch_{epoch}.pth"))
    #         torch.save(model.state_dict(), uncertainty_model_path_epoch)
    # torch.save(model.state_dict(), uncertainty_model_path)

def evaluate_classification_compound_with_uncertainty_model(model, data_loader, device, mode='test'):
    model.eval()
    total_class_loss = 0
    total_epi_loss = 0
    correct = 0
    all_sph_err = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            outputs, mid_value, sph_err = model(data)
            class_loss = F.cross_entropy(outputs[:,:,0,0], labels)
            epi_loss = torch.mean(sph_err,dim=(1,2,3)) + torch.mean((mid_value-outputs)**2,dim=(1,2,3))
            total_class_loss += class_loss.item()
            total_epi_loss += torch.mean(epi_loss).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            all_sph_err.append(epi_loss.detach().cpu().numpy())
    all_sph_err = np.concatenate(all_sph_err, axis=0)
    if mode == 'test':
        np.savez(uncertainty_test_outputs_path, sph_err=all_sph_err)
    elif mode == 'valid':
        np.savez(uncertainty_valid_outputs_path, sph_err=all_sph_err)
    elif mode == 'train':
        np.savez(uncertainty_train_outputs_path, sph_err=all_sph_err)
    print(f'{mode}, Class Loss: {total_class_loss / (batch_idx+1):.8f}')
    print(f'{mode}, Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
    print(f'{mode}, Class Accuracy: {correct / (len(data_loader.dataset)):.4f}')
    return all_sph_err

if __name__ == '__main__':
    ## load datasets
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size, fast_test=fast_test)
    ### train eaae model
    if uncertainty_model_name == 'eaFNO':
        uncertainty_model = EABlockFNO(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaCNN':
        uncertainty_model = EABlockCNN(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaae':
        uncertainty_model = EAAE(batch_size,device,im_x, im_y, hidden_dim, latent_dim, input_channels,num_classes, modes1, modes2)
    elif uncertainty_model_name == 'compound':
        uncertainty_model = CompoundModel(im_x, im_y, hidden_dim, num_classes, input_channels, modes1, modes2)
    ## if trained uncertainty model exists, load it
    if uncertainty_model_initial_path.exists() and use_old_uncertainty_model:
        print(f"Loading trained uncertainty model from {uncertainty_model_initial_path}")
        uncertainty_model.load_state_dict(torch.load(uncertainty_model_initial_path))
    else:
        print("No trained uncertainty model found, training a new model.")
    uncertainty_model = uncertainty_model.to(device)
    optimizer = Adam(uncertainty_model.parameters(), lr=uncertainty_lr)
    begin_epoch = 1
    best_loss = float('inf')
    for epoch in range(begin_epoch, uncertainty_epochs+begin_epoch):
        total_class_loss, correct, total_epi_loss = train_classification_compound_with_uncertainty_model(uncertainty_model, optimizer, train_loader, device)
        if epoch % log_interval == 0:
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Class Loss: {total_class_loss:.8f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Epi Loss: {total_epi_loss:.8f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Class Accuracy: {correct:.4f}')
            # # Log epoch results
            # result_logger.log({
            #     'ep': epoch,
            #     'train_class_loss': total_class_loss,
            #     'train_correct': correct,
            #     'train_epi_loss': total_epi_loss
            # })
            # total_loss = total_class_loss + total_epi_loss
            # if total_loss < best_loss:
            #     best_loss = total_loss
            #     torch.save(uncertainty_model.state_dict(), uncertainty_model_path)
            # ### use the stored best model if the current epoch model is not the best
            # else:
            #     uncertainty_model.load_state_dict(torch.load(uncertainty_model_path))

    train_uncertainty = evaluate_classification_compound_with_uncertainty_model(uncertainty_model, train_loader, device, mode='train')
    test_uncertainty = evaluate_classification_compound_with_uncertainty_model(uncertainty_model, test_loader, device, mode='test')
    valid_uncertainty = evaluate_classification_compound_with_uncertainty_model(uncertainty_model, valid_loader, device, mode='valid')
    train_scores = np.exp(-train_uncertainty)
    test_scores = np.exp(-test_uncertainty)
    valid_scores = np.exp(-valid_uncertainty)
    metrics_train_test = compute_ood_metrics(train_scores, test_scores)
    metrics_train_valid = compute_ood_metrics(train_scores, valid_scores)
    metrics_test_valid = compute_ood_metrics(test_scores, valid_scores)
    OOD_result_logger.log({
        'ep': epoch,
        'train_vs_test': metrics_train_test['AUROC'],
        'train_vs_valid': metrics_train_valid['AUROC'],
        'test_vs_valid': metrics_test_valid['AUROC']
    })
