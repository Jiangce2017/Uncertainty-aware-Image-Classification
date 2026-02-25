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
    hidden_dim= 64
    x_dim = im_x*im_y*input_channels

    vae_hidden_dim = 64
    vae_latent_dim = 32
    
# Default hyperparameters
vae_lr = 1e-4   
uncertainty_lr = 1e-4
log_interval = 10
classification_epochs = 5
vae_epochs = 200
uncertainty_epochs = 1000
batch_size = 64
data_root = '/ocean/projects/cis240139p/jchen39/datasets' 
results_dir = Path("/jet/home/jchen39/projects/EAAE/results10")
models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints10")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

fast_test = False  # Set to True for faster benchmarking with a smaller dataset
use_old_classification_model = False
use_old_classification_data = False
use_old_vae_model = False
use_old_vae_data = False
use_old_uncertainty_model = False

classification_model_name = 'classification_cnn'
vae_model_name = 'vaecnn' #'vaecnn' # 'freqfno' 
uncertainty_model_name = 'compound' #'eaCNN', 'eaae', 'eaFNO'

experiment_name = dataset_name+ "_"+ str(hidden_dim) + "_" + str(latent_dim)+"_"+str(modes1)+"_"+str(modes2)

classification_model_path = Path(os.path.join(models_dir, dataset_name + '_classification.pth'))
if not fast_test:
    classification_train_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_train_outputs.npz'))
    classification_test_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_test_outputs.npz'))
    classification_valid_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_valid_outputs.npz'))    
else:
    classification_train_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_train_outputs_fast.npz'))
    classification_test_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_test_outputs_fast.npz'))
    classification_valid_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + dataset_name + '_classification_valid_outputs_fast.npz'))
vae_model_path = Path(os.path.join(models_dir, vae_model_name+ "_" + experiment_name + ".pth"))

vae_train_outputs_path = Path(os.path.join(data_root, vae_model_name+ "_" + experiment_name + '_vae_train_outputs.npz'))
vae_test_outputs_path = Path(os.path.join(data_root, vae_model_name+ "_" + experiment_name + '_vae_test_outputs.npz'))
vae_valid_outputs_path = Path(os.path.join(data_root, vae_model_name+ "_" + experiment_name + '_vae_valid_outputs.npz'))

uncertainty_model_path = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name +".pth"))

uncertainty_train_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_train_outputs.npz'))
uncertainty_test_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_test_outputs.npz'))
uncertainty_valid_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_valid_outputs.npz'))

uncertainty_vae_train_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_vae_train_outputs.npz'))
uncertainty_vae_test_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_vae_test_outputs.npz'))
uncertainty_vae_valid_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name + '_uncertainty_vae_valid_outputs.npz'))

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
        train_dataset = torch.utils.data.Subset(train_dataset, range(128))
        # test_dataset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), size=128, replace=False))
        # valid_dataset = torch.utils.data.Subset(valid_dataset, np.random.choice(len(valid_dataset), size=128, replace=False))
    test_dataset = torch.utils.data.Subset(test_dataset, range(128))  # Use a subset for faster benchmarking
    valid_dataset = torch.utils.data.Subset(valid_dataset, range(128))  # Use a subset for faster benchmarking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader, test_loader, valid_loader

def train_classification_model(model, train_loader, device):
    optimizer = Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(1, classification_epochs+1):
        total_loss = 0
        correct = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, features = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            ### calculate the classification accuracy for this batch
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total_loss += loss.item()
        if epoch % log_interval == 0:
            print(f'Epoch [{epoch}/{classification_epochs}], Class Loss: {total_loss / (batch_idx+1):.8f}')
            print(f'Epoch [{epoch}/{classification_epochs}], Class Accuracy: {correct / (len(train_loader.dataset)):.4f}')
            torch.save(model.state_dict(), classification_model_path)

    print("Training complete and model saved.")
    torch.save(model.state_dict(), classification_model_path)

def evaluate_classification_model(model, train_loader, device, mode='test'):
    model.eval()
    all_data = []
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            outputs, features = model(data)
            #all_outputs.append(outputs.cpu().numpy())
            all_outputs.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_data.append(data.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_data = np.concatenate(all_data, axis=0)
    if mode == 'test':
        np.savez(classification_test_outputs_path, probs=all_outputs, labels=all_labels, data=all_data)
    elif mode == 'valid':
        np.savez(classification_valid_outputs_path, probs=all_outputs, labels=all_labels, data=all_data)
    elif mode == 'train':
        np.savez(classification_train_outputs_path, probs=all_outputs, labels=all_labels, data=all_data)

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

def train_uncertainty_model_with_classification(uncertainty_model, classification_data_loader):
    optimizer = Adam(uncertainty_model.parameters(), lr=uncertainty_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=uncertainty_epochs*len(classification_data_loader))
    uncertainty_model.train()
    for epoch in range(1, uncertainty_epochs+1):
        total_epi_loss = 0
        total_mid_value_loss = 0
        correct = 0
        for batch_idx, (data, outputs, labels) in enumerate(classification_data_loader):
            data, outputs, labels = data.to(device), outputs.to(device), labels.to(device)
            #outputs = torch.softmax(outputs, dim=1)+1
            #print("outputs shape:", outputs.shape)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

            #outputs = torch.sigmoid(outputs)*0.25+0.25
            #outputs = torch.softmax(outputs,dim=1)*0.25+0.25
            outputs = outputs[:,:,None,None]
            #print("Train, max output:", torch.max(outputs).item(), "min output:", torch.min(outputs).item())
            optimizer.zero_grad()
            if uncertainty_model_name == 'eaae':
                #outputs = torch.sigmoid(outputs)*0.25+0.25
                mid_value, sph_err_encoder, mid_value_decoder, sph_err_decoder = uncertainty_model(data,outputs)
                epi_loss = torch.mean(sph_err_decoder) + torch.mean(sph_err_encoder)
                #mid_value_decoder = torch.sigmoid(mid_value_decoder)
                mid_value_loss = torch.mean((mid_value_decoder-outputs)**2) #+ torch.mean((mid_value_decoder-mid_value)**2)
            else:
                mid_value, sph_err = uncertainty_model(data,outputs)
                epi_loss = torch.mean(sph_err) 
                mid_value_loss = torch.mean((mid_value-outputs)**2)
            loss = epi_loss + mid_value_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_epi_loss += epi_loss.item()
            total_mid_value_loss += mid_value_loss.item()
        if epoch % log_interval == 0:
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Class Accuracy: {correct / (len(classification_data_loader.dataset)):.4f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Mid Value Loss: {total_mid_value_loss / (batch_idx+1):.8f}')
            torch.save(uncertainty_model.state_dict(), uncertainty_model_path)
            # Log epoch results
            result_logger.log({
                'ep': epoch,
                'train_epi_loss': total_epi_loss / (batch_idx+1),
                'train_mid_value_loss': total_mid_value_loss / (batch_idx+1)
            })
    print("Training complete and model saved.")
    torch.save(uncertainty_model.state_dict(), uncertainty_model_path)

def evaluate_uncertainty_model_with_classification(uncertainty_model, classification_data_loader, mode='test'):
    uncertainty_model.eval()
    classify_model.eval()
    total_epi_loss = 0
    total_mid_value_loss = 0
    all_sph_err = []
    with torch.no_grad():
        for batch_idx, (data, outputs, labels) in enumerate(classification_data_loader):
            data, outputs, labels = data.to(device), outputs.to(device), labels.to(device)
            #outputs = classify_model(data)
            ## mask the correct prediction
            pred = outputs.argmax(dim=1, keepdim=True)
            correct_mask = labels.eq(pred.view_as(labels))
            #outputs = torch.sigmoid(outputs)*0.25+0.25
            #outputs = torch.softmax(outputs,dim=1)*0.25+0.25
            outputs = outputs[:,:,None,None]
            if uncertainty_model_name == 'eaae':
                outputs = torch.sigmoid(outputs)*0.25+0.25
                mid_value, sph_err_encoder, mid_value_decoder, sph_err_decoder = uncertainty_model(data,outputs)
                epi_loss = torch.mean(sph_err_decoder,dim=(1,2,3)) + torch.mean(sph_err_encoder,dim=(1,2,3))
                mid_value_loss = torch.mean((mid_value_decoder-outputs)**2,dim=(1,2,3)) #+ torch.mean((mid_value_decoder-mid_value)**2,dim=(1,2,3))
            else:
                mid_value, sph_err = uncertainty_model(data,outputs)
                epi_loss = torch.mean(sph_err, dim=(1,2,3)) 
                mid_value_loss = torch.mean((mid_value-outputs)**2, dim=(1,2,3))
            # if mode == 'test' or mode == 'train':
            #     epi_loss = epi_loss[correct_mask]
            #     mid_value_loss = mid_value_loss[correct_mask]
            total_epi_loss += torch.mean(epi_loss).item()
            total_mid_value_loss += torch.mean(mid_value_loss).item()
            all_sph_err.append((epi_loss+mid_value_loss).detach().cpu().numpy())
    all_sph_err = np.concatenate(all_sph_err, axis=0)
    if mode == 'test':
        np.savez(uncertainty_test_outputs_path, sph_err=all_sph_err)
    elif mode == 'valid':
        np.savez(uncertainty_valid_outputs_path, sph_err=all_sph_err)
    elif mode == 'train':
        np.savez(uncertainty_train_outputs_path, sph_err=all_sph_err)
    print(f'Test Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
    print(f'Test Mid Value Loss: {total_mid_value_loss / (batch_idx+1):.8f}')
    return all_sph_err

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross-Entropy for pixel values 0-1) [2]
    # print("recon_x shape:", recon_x.shape)
    # print("x shape:", x.shape)
    BCE = F.binary_cross_entropy(recon_x.view(-1, x_dim), x.view(-1, x_dim), reduction='mean')
    # KL divergence loss (penalizes deviation from standard normal distribution) [2]
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE# + KLD

def train_vae_model(model, train_loader):
    # Initialize model, optimizer
    optimizer = Adam(model.parameters(), lr=vae_lr)
    model.train()
    for epoch in range(1,vae_epochs+1):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            if vae_model_name == 'vaecnn':
                loss = vae_loss(recon_batch, data, mu, logvar)
            elif vae_model_name == 'freqfno':
                loss = F.binary_cross_entropy(recon_batch.view(-1, x_dim), data.view(-1, x_dim), reduction='mean')+torch.mean(logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{vae_epochs}], Loss: {total_loss / (batch_idx+1):.8f}')
            torch.save(model.state_dict(), vae_model_path)
            # Log epoch results
            vae_result_logger.log({
                'ep': epoch,
                'vae_loss': total_loss / (batch_idx+1)
            })

    # Save the model
    torch.save(model.state_dict(), vae_model_path)
    print("Training complete and model saved.")
    # return all_mu, all_data

def evaluate_vae_model(model, data_loader, device, mode='test'):
    model.eval()
    all_mu = []
    all_data = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            mu, _ = model.Encoder(data)
            all_mu.append(mu.detach().cpu().numpy())
            all_data.append(data.detach().cpu().numpy())
        all_mu = np.concatenate(all_mu, axis=0)
        all_data = np.concatenate(all_data, axis=0)
    if mode == 'test':
        np.savez(vae_test_outputs_path, mu=all_mu, data=all_data)
    elif mode == 'valid':
        np.savez(vae_valid_outputs_path, mu=all_mu, data=all_data)
    elif mode == 'train':
        np.savez(vae_train_outputs_path, mu=all_mu, data=all_data)

def train_uncertainty_model(uncertainty_model, vae_data_loader):
    print(f"Uncertainty Model: {uncertainty_model_name}")
    print(f"EA Block Parameters: {sum(p.numel() for p in uncertainty_model.parameters()):,}")
    optimizer = Adam(uncertainty_model.parameters(), lr=uncertainty_lr)
    uncertainty_model.train()
    vae_model.eval()
    for epoch in range(1, uncertainty_epochs+1):
        total_epi_loss = 0
        total_mid_value_loss = 0
        total_vae_loss = 0
        for batch_idx, (data, mu) in enumerate(vae_data_loader):
            data, mu = data.to(device), mu.to(device)
            optimizer.zero_grad()
            # recon_batch, mu, logvar = vae_model(data)
            # vae_loss_value = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            # mu = (mu - train_mu_mean) / train_mu_std
            # mu = mu - train_mu_min + 0.1
            # mu = torch.sigmoid(mu)+1
            mu = mu[:,:,None,None]
            if uncertainty_model_name == 'eaae':
                mid_value, sph_err_encoder, mid_value_decoder, sph_err_decoder = uncertainty_model(data)
                epi_loss = torch.mean(sph_err_decoder) + torch.mean(sph_err_encoder)
                mid_value_loss = torch.mean((mid_value_decoder-mu)**2)
            else:
                mid_value, sph_err = uncertainty_model(data,mu)
                epi_loss = torch.mean(sph_err) 
                mid_value_loss = torch.mean((mid_value-mu)**2)
            loss = epi_loss + mid_value_loss
            loss.backward()
            optimizer.step()
            total_epi_loss += epi_loss.item()
            total_mid_value_loss += mid_value_loss.item()
            #total_vae_loss += vae_loss_value.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
            print(f'Epoch [{epoch}/{uncertainty_epochs}], Mid Value Loss: {total_mid_value_loss / (batch_idx+1):.8f}')
            #print(f'Epoch [{epoch}/{uncertainty_epochs}], VAE Loss: {total_vae_loss / (batch_idx+1):.8f}')
            # Log epoch results
            result_logger.log({
                'ep': epoch,
                'train_epi_loss': total_epi_loss / (batch_idx+1),
                'train_mid_value_loss': total_mid_value_loss / (batch_idx+1)
            })
            torch.save(uncertainty_model.state_dict(), uncertainty_model_path)
    # Save the model
    torch.save(uncertainty_model.state_dict(), uncertainty_model_path)
    print("Training complete and model saved.")
    #return total_epi_loss, all_sph_err

def evaluate_uncertainty_model(uncertainty_model, vae_data_loader,mode='test'):
    uncertainty_model.eval()
    vae_model.eval()
    total_epi_loss = 0
    total_mid_value_loss = 0
    all_sph_err = []
    with torch.no_grad():
        for batch_idx, (data, mu) in enumerate(vae_data_loader):
            data, mu = data.to(device), mu.to(device)
            #vae_loss_value = F.binary_cross_entropy(recon_batch, data, reduce=None)
            #vae_loss_value = torch.mean((recon_batch-data)**2, dim=(1,2,3))
            # mu = (mu - train_mu_mean) / train_mu_std
            # mu = mu - train_mu_min + 0.1
            #mu = torch.sigmoid(mu)+1
            mu = mu[:,:,None,None]
            if uncertainty_model_name == 'eaae':
                mid_value, sph_err_encoder, mid_value_decoder, sph_err_decoder = uncertainty_model(data)
                epi_loss = torch.mean(sph_err_decoder, dim=(1,2,3)) + torch.mean(sph_err_encoder, dim=(1,2,3))
                mid_value_loss = torch.mean((mid_value_decoder-mu)**2, dim=(1,2,3)) #+ torch.mean((mid_value_decoder-mid_value)**2, dim=(1,2,3))
            else:
                mid_value, sph_err = uncertainty_model(data,mu)
                epi_loss = torch.mean(sph_err, dim=(1,2,3))
                mid_value_loss = torch.mean((mid_value-mu)**2, dim=(1,2,3))
            total_loss =epi_loss + mid_value_loss #+ vae_loss_value
            all_sph_err.append(total_loss.detach().cpu().numpy())
            total_epi_loss += torch.mean(epi_loss).item()
            total_mid_value_loss += torch.mean(mid_value_loss).item()
            #total_vae_loss += torch.mean(vae_loss_value).item()
    print(f'Test Epi Loss: {total_epi_loss / (batch_idx+1):.8f}')
    print(f'Test Mid Value Loss: {total_mid_value_loss / (batch_idx+1):.8f}')
    #print(f'Test VAE Loss: {total_vae_loss / (batch_idx+1):.8f}')
    all_sph_err = np.concatenate(all_sph_err, axis=0)
    if mode == 'test':
        np.savez(uncertainty_vae_test_outputs_path, sph_err=all_sph_err)
    elif mode == 'valid':
        np.savez(uncertainty_vae_valid_outputs_path, sph_err=all_sph_err)
    elif mode == 'train':
        np.savez(uncertainty_vae_train_outputs_path, sph_err=all_sph_err)
    return all_sph_err

def evaluate_vae_reconstruction(vae_model, data_loader, mode='train'):
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
        image_name = "reconstruction_" + vae_model_name + "_"+ experiment_name + "_" + mode + "_" + str(batch_idx) + ".png"
        plt.savefig(osp.join(results_dir, image_name))
        plt.close(fig)

def load_classification_data(classification_data, batch_size,mode='train',data_normalizer=None, output_normalizer=None):
    all_classification_outputs = classification_data['probs']
    all_classification_labels = classification_data['labels']
    all_classification_data = classification_data['data']
    ## create a new dataloader for uncertainty training
    classification_outputs_tensor = torch.tensor(all_classification_outputs, dtype=torch.float32)
    classification_labels_tensor = torch.tensor(all_classification_labels, dtype=torch.long)
    classification_data_tensor = torch.tensor(all_classification_data, dtype=torch.float32)
    if mode == 'train':
        data_normalizer = GaussianNormalizer(classification_data_tensor)
        output_normalizer = GaussianNormalizer(classification_outputs_tensor)
        classification_data_tensor = data_normalizer.encode(classification_data_tensor)
        classification_outputs_tensor = output_normalizer.encode(classification_outputs_tensor)
    else:
        classification_data_tensor = data_normalizer.encode(classification_data_tensor)
        classification_outputs_tensor = output_normalizer.encode(classification_outputs_tensor)

    classification_datasets = torch.utils.data.TensorDataset(classification_data_tensor, classification_outputs_tensor,classification_labels_tensor)
    ### randomly select a subset of the data for uncertainty model training if fast_test is True
    # subset_indices = np.random.choice(len(classification_datasets), size=128, replace=False)
    # classification_datasets = torch.utils.data.Subset(classification_datasets, subset_indices)
    classification_data_loader = DataLoader(classification_datasets, batch_size=batch_size, shuffle=True)
    if mode == 'train':
        return classification_data_loader, data_normalizer, output_normalizer
    else:
        return classification_data_loader

def load_vae_data(vae_data, batch_size,mode='train',data_normalizer=None, output_normalizer=None):
    all_vae_outputs = vae_data['mu']
    all_vae_data = vae_data['data']
    ## create a new dataloader for uncertainty training
    vae_outputs_tensor = torch.tensor(all_vae_outputs, dtype=torch.float32)
    vae_data_tensor = torch.tensor(all_vae_data, dtype=torch.float32)
    if mode == 'train':
        data_normalizer = RangeNormalizer(vae_data_tensor)
        output_normalizer = RangeNormalizer(vae_outputs_tensor)
        vae_data_tensor = data_normalizer.encode(vae_data_tensor)
        vae_outputs_tensor = output_normalizer.encode(vae_outputs_tensor)
    else:
        vae_data_tensor = data_normalizer.encode(vae_data_tensor)
        vae_outputs_tensor = output_normalizer.encode(vae_outputs_tensor)

    vae_datasets = torch.utils.data.TensorDataset(vae_data_tensor, vae_outputs_tensor)
    ### randomly select a subset of the data for uncertainty model training if fast_test is True
    # subset_indices = np.random.choice(len(vae_datasets), size=128, replace=False)
    # vae_datasets = torch.utils.data.Subset(vae_datasets, subset_indices)
    vae_data_loader = DataLoader(vae_datasets, batch_size=batch_size, shuffle=True)
    if mode == 'train':
        return vae_data_loader, data_normalizer, output_normalizer
    else:
        return vae_data_loader

if __name__ == '__main__':
    ## load datasets
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size, fast_test=fast_test)
    
    # ### train classification model
    # block = Bottleneck
    # num_blocks = [3, 4, 6, 3]
    # classify_model = ResNet(block, num_blocks,input_channels=input_channels, num_classes=num_classes).to(device)
    # if classification_model_path.exists() and use_old_classification_model:
    #     print(f"Loading trained classification model from {classification_model_path}")
    #     classify_model.load_state_dict(torch.load(classification_model_path))
    # else:
    #     print("No trained classification model found, training a new model.")
        
    # if not use_old_classification_model:
    #     train_classification_model(classify_model, train_loader, device)

    # if not use_old_classification_data:
    #     evaluate_classification_model(classify_model, train_loader, device, mode='train')
    #     evaluate_classification_model(classify_model, test_loader, device, mode='test')
    #     evaluate_classification_model(classify_model, valid_loader, device, mode='valid')
    # classification_train_data = np.load(classification_train_outputs_path)
    # classification_test_data = np.load(classification_test_outputs_path)
    # classification_valid_data = np.load(classification_valid_outputs_path)
    # classification_train_loader, data_normalizer, output_normalizer = load_classification_data(classification_train_data, batch_size,mode='train')
    # classification_test_loader = load_classification_data(classification_test_data, batch_size,mode='test',data_normalizer=data_normalizer, output_normalizer=output_normalizer)
    # classification_valid_loader = load_classification_data(classification_valid_data, batch_size,mode='valid',data_normalizer=data_normalizer, output_normalizer=output_normalizer)
    
    
    # ### train vae model
    # if vae_model_name == 'vaecnn':
    #     vae_model = VAE_CNN(input_channels, output_channels, vae_hidden_dim, vae_latent_dim,device,im_x,im_y).to(device)
    # elif vae_model_name == 'freqfno':
    #     vae_model = Freq_FNO(batch_size, device, input_channels, vae_latent_dim, vae_hidden_dim, im_x, im_y, modes1, modes2).to(device)
    # if vae_model_path.exists() and use_old_vae_model:
    #     print(f"Loading trained VAE model from {vae_model_path}")
    #     vae_model.load_state_dict(torch.load(vae_model_path))
    # else:
    #     print("No trained VAE model found, training a new model.")

    # if not use_old_vae_model:
    #     train_vae_model(vae_model, train_loader)
    # vae_model.eval()
    # evaluate_vae_reconstruction(vae_model, train_loader, mode='train')
    # evaluate_vae_reconstruction(vae_model, test_loader, mode='test')
    # evaluate_vae_reconstruction(vae_model, valid_loader, mode='valid')
    
    # if not use_old_vae_data:
    #     evaluate_vae_model(vae_model, train_loader, device, mode='train')
    #     evaluate_vae_model(vae_model, test_loader, device, mode='test')
    #     evaluate_vae_model(vae_model, valid_loader, device, mode='valid')

    # vae_train_data = np.load(vae_train_outputs_path)
    # vae_test_data = np.load(vae_test_outputs_path)
    # vae_valid_data = np.load(vae_valid_outputs_path)
    # vae_train_loader, vae_data_normalizer, vae_output_normalizer = load_vae_data(vae_train_data, batch_size,mode='train')
    # vae_test_loader = load_vae_data(vae_test_data, batch_size,mode='test',data_normalizer=vae_data_normalizer, output_normalizer=vae_output_normalizer)
    # vae_valid_loader = load_vae_data(vae_valid_data, batch_size,mode='valid',data_normalizer=vae_data_normalizer, output_normalizer=vae_output_normalizer)

    ### train eaae model
    if uncertainty_model_name == 'eaFNO':
        uncertainty_model = EABlockFNO(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
        #uncertainty_model = EABlockFNO(im_x, im_y, hidden_dim, vae_latent_dim, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaCNN':
        uncertainty_model = EABlockCNN(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaae':
        uncertainty_model = EAAE(batch_size,device,im_x, im_y, hidden_dim, latent_dim, input_channels,num_classes, modes1, modes2)
    elif uncertainty_model_name == 'compound':
        uncertainty_model = CompoundModel(im_x, im_y, hidden_dim, num_classes, input_channels, modes1, modes2)
    ## if trained uncertainty model exists, load it
    if uncertainty_model_path.exists() and use_old_uncertainty_model:
        print(f"Loading trained uncertainty model from {uncertainty_model_path}")
        uncertainty_model.load_state_dict(torch.load(uncertainty_model_path))
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
            # Log epoch results
            result_logger.log({
                'ep': epoch,
                'train_class_loss': total_class_loss,
                'train_correct': correct,
                'train_epi_loss': total_epi_loss
            })
            total_loss = total_class_loss + total_epi_loss
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(uncertainty_model.state_dict(), uncertainty_model_path)

            # uncertainty_model_path_epoch = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name + f"_epoch_{epoch}.pth"))
            # torch.save(uncertainty_model.state_dict(), uncertainty_model_path_epoch)
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

    # train_uncertainty_model_with_classification(uncertainty_model, classification_train_loader)
    # train_uncertainty = evaluate_uncertainty_model_with_classification(uncertainty_model,classification_train_loader,mode='train')
    # test_uncertainty = evaluate_uncertainty_model_with_classification(uncertainty_model,classification_test_loader,mode='test')
    # valid_uncertainty = evaluate_uncertainty_model_with_classification(uncertainty_model,classification_valid_loader,mode='valid')

    # train_uncertainty_model(uncertainty_model, vae_train_loader)
    # train_uncertainty = evaluate_uncertainty_model(uncertainty_model,vae_train_loader,mode='train')
    # test_uncertainty = evaluate_uncertainty_model(uncertainty_model,vae_test_loader,mode='test')
    # valid_uncertainty = evaluate_uncertainty_model(uncertainty_model,vae_valid_loader,mode='valid')


    # print("Train mean uncertainty:", np.mean(train_uncertainty))
    # print("Test mean uncertainty:", np.mean(test_uncertainty))
    # print("Valid mean uncertainty:", np.mean(valid_uncertainty))
    # print("Train median uncertainty:", np.median(train_uncertainty))
    # print("Test median uncertainty:", np.median(test_uncertainty))
    # print("Valid median uncertainty:", np.median(valid_uncertainty))
    # print("Train min uncertainty:", np.min(train_uncertainty))
    # print("Train max uncertainty:", np.max(train_uncertainty))
    # print("Test min uncertainty:", np.min(test_uncertainty))
    # print("Test max uncertainty:", np.max(test_uncertainty))
    # print("Valid min uncertainty:", np.min(valid_uncertainty))
    # print("Valid max uncertainty:", np.max(valid_uncertainty))
    # train_scores = np.exp(-train_uncertainty)
    # test_scores = np.exp(-test_uncertainty)
    # valid_scores = np.exp(-valid_uncertainty)
    # print("train_scores shape:", train_scores.shape)
    # print("test_scores shape:", test_scores.shape)
    # print("valid_scores shape:", valid_scores.shape)
    # metrics_train_test = compute_ood_metrics(train_scores, test_scores)
    # print("OOD Detection Metrics between train and test:", metrics_train_test)
    # metrics_train_valid = compute_ood_metrics(train_scores, valid_scores)
    # print("OOD Detection Metrics between train and valid:", metrics_train_valid)
    # metrics_test_valid = compute_ood_metrics(test_scores, valid_scores)
    # print("OOD Detection Metrics between test and valid:", metrics_test_valid)
    # ### save the above results to in a text file
    # with open(out_of_distribution_results_path, 'w') as f:
    #     f.write(f"Train mean uncertainty: {np.mean(train_uncertainty)}\n")
    #     f.write(f"Test mean uncertainty: {np.mean(test_uncertainty)}\n")
    #     f.write(f"Valid mean uncertainty: {np.mean(valid_uncertainty)}\n")
    #     f.write(f"Train median uncertainty: {np.median(train_uncertainty)}\n")
    #     f.write(f"Test median uncertainty: {np.median(test_uncertainty)}\n")
    #     f.write(f"Valid median uncertainty: {np.median(valid_uncertainty)}\n")
    #     f.write(f"Train min uncertainty: {np.min(train_uncertainty)}\n")
    #     f.write(f"Train max uncertainty: {np.max(train_uncertainty)}\n")
    #     f.write(f"Test min uncertainty: {np.min(test_uncertainty)}\n")
    #     f.write(f"Test max uncertainty: {np.max(test_uncertainty)}\n")
    #     f.write(f"Valid min uncertainty: {np.min(valid_uncertainty)}\n")
    #     f.write(f"Valid max uncertainty: {np.max(valid_uncertainty)}\n")
    #     f.write(f"OOD Detection Metrics between train and test: {metrics_train_test}\n")
    #     f.write(f"OOD Detection Metrics between train and valid: {metrics_train_valid}\n")
    #     f.write(f"OOD Detection Metrics between test and valid: {metrics_test_valid}\n")
    # print(f"Results saved to: {out_of_distribution_results_path}")   