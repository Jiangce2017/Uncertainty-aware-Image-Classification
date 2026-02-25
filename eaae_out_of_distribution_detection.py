### evaluate the performance of the uncertainty estimation methods on out-of-distribution detection. 
#We use the same model trained on the in-distribution dataset, 
#and evaluate the OOD detection performance on a different dataset. 
#For example, if the model is trained on CIFAR-10, we can evaluate the OOD detection performance on SVHN or CIFAR-100. 
#We can use the same metrics as above to evaluate the OOD detection performance, such as AUROC, AUPR, FPR at 95% TPR, etc. 
#We can also compare the OOD detection performance of different uncertainty estimation methods, such as softmax confidence, entropy, mutual information, etc.

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import time
import json
from pathlib import Path
import os
import os.path as osp
import csv

from utils import compute_ood_metrics

# from model import get_model, MODELS
from model import EABlockFNO, EABlockCNN,CompoundModel
from eaae_model import EAAE
from vae import VAE_CNN
from resnet import BasicBlock, Bottleneck, ResNet 

# Global parameters
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  # Device to use: 'cpu' or 'cuda'

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
    epi_channels=10
    latent_dim=64
    modes1 = 10
    modes2 = 6
    num_classes = 10
    hidden_dim=128
    x_dim = im_x*im_y*input_channels
    
# Default hyperparameters
batch_size = 64
data_root = '/ocean/projects/cis240139p/jchen39/datasets' 
results_dir = Path("/jet/home/jchen39/projects/EAAE/results10")
models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints10")

classification_model_name = 'classification_cnn'
vae_model_name = 'vaecnn'
uncertainty_model_name = 'compound' #'eaCNN', 'eaae', 'eaFNO'

experiment_name = dataset_name+ "_"+ str(hidden_dim) + "_" + str(latent_dim)+"_"+str(modes1)+"_"+str(modes2)

classification_model_path = Path(os.path.join(models_dir, dataset_name + '_classification.pth'))
classification_train_outputs_path = Path(os.path.join(data_root, classification_model_name+ "_" + experiment_name + '_classification_train_outputs.npz'))    
vae_model_path = Path(os.path.join(models_dir, vae_model_name+ "_" + experiment_name + ".pth"))
uncertainty_model_path = Path(os.path.join(models_dir, uncertainty_model_name+ "_" + experiment_name +"_best.pth"))
uncertainty_train_outputs_path = Path(os.path.join(data_root, uncertainty_model_name+ "_" + experiment_name  + '_uncertainty_train_outputs.npz'))
out_of_distribution_results_path = os.path.join(results_dir, uncertainty_model_name+ "_" + experiment_name +'_stat_results.txt')

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_loader, test_loader, valid_loader


def evaluate_ood_detection(model, in_loader, out_loader, device):
    model.eval()
    in_uncertainty = []
    out_uncertainty = []
    total_class_loss = 0
    total_epi_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(in_loader):
            data, labels = data.to(device), labels.to(device)
            outputs, mid_value, sph_err = model(data)
            class_loss = F.cross_entropy(outputs[:,:,0,0], labels)
            pred = outputs[:,:,0,0].argmax(dim=1, keepdim=True)
            correct_mask = labels.eq(pred.view_as(labels))
            epi_loss = torch.mean(sph_err[correct_mask],dim=(1,2,3)) + torch.mean((mid_value[correct_mask]-outputs[correct_mask])**2,dim=(1,2,3))
            total_class_loss += class_loss.item()
            total_epi_loss += torch.mean(epi_loss).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            in_uncertainty.append(epi_loss.detach().cpu().numpy())
        print("Average epi_loss on in-distribution data:", total_epi_loss / (batch_idx+1))
        print("Accuracy on in-distribution data:", correct / len(in_loader.dataset))
        
        total_class_loss = 0
        total_epi_loss = 0
        correct = 0
        
        for batch_idx, (data, labels) in enumerate(out_loader):
            data, labels = data.to(device), labels.to(device)
            outputs, mid_value, sph_err = model(data)
            class_loss = F.cross_entropy(outputs[:,:,0,0], labels)
            epi_loss = torch.mean(sph_err,dim=(1,2,3)) + torch.mean((mid_value-outputs)**2,dim=(1,2,3))
            total_class_loss += class_loss.item()
            total_epi_loss += torch.mean(epi_loss).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            out_uncertainty.append(epi_loss.detach().cpu().numpy())
        print("Average epi_loss on out-of-distribution data:", total_epi_loss / (batch_idx+1))
        print("Accuracy on out-of-distribution data:", correct / len(out_loader.dataset))
    
    in_uncertainty = np.concatenate(in_uncertainty, axis=0)
    out_uncertainty = np.concatenate(out_uncertainty, axis=0)
    ## print the shape of in_uncertainty and out_uncertainty
    print(f"Shape of in-distribution uncertainty: {in_uncertainty.shape}")
    print(f"Shape of out-of-distribution uncertainty: {out_uncertainty.shape}")
    ### calcuate the mean and median of the uncertainty scores for in-distribution and out-of-distribution samples
    in_uncertainty_mean = np.mean(in_uncertainty)
    out_uncertainty_mean = np.mean(out_uncertainty)
    in_uncertainty_median = np.median(in_uncertainty)
    out_uncertainty_median = np.median(out_uncertainty)
    print(f"In-distribution uncertainty mean: {in_uncertainty_mean}, median: {in_uncertainty_median}")
    print(f"Out-of-distribution uncertainty mean: {out_uncertainty_mean}, median: {out_uncertainty_median}")
    ### calcuate the 10-percentile and 90-percentile of the uncertainty scores for in-distribution and out-of-distribution samples
    in_uncertainty_10_percentile = np.percentile(in_uncertainty, 10)
    out_uncertainty_10_percentile = np.percentile(out_uncertainty, 10)
    in_uncertainty_90_percentile = np.percentile(in_uncertainty, 90)
    out_uncertainty_90_percentile = np.percentile(out_uncertainty, 90)
    print(f"In-distribution uncertainty 10-percentile: {in_uncertainty_10_percentile}, 90-percentile: {in_uncertainty_90_percentile}")
    print(f"Out-of-distribution uncertainty 10-percentile: {out_uncertainty_10_percentile}, 90-percentile: {out_uncertainty_90_percentile}")
    ### set the threshold for OOD detection based on the 90-percentile of the in-distribution uncertainty scores
    in_scores = np.exp(-in_uncertainty)  # Higher uncertainty should correspond to lower confidence
    out_scores = np.exp(-out_uncertainty)  # Higher uncertainty should correspond to
    # Compute OOD detection metrics (e.g., AUROC, AUPR, FPR at 95% TPR)
    # This is a placeholder for the actual metric computation
    metrics = compute_ood_metrics(in_scores, out_scores)

    ### draw the histogram of the uncertainty scores for in-distribution and out-of-distribution samples
    import matplotlib.pyplot as plt
    plt.hist(in_scores, bins=50, alpha=0.5, label='In-Distribution')
    plt.hist(out_scores, bins=50, alpha=0.5, label='Out-of-Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Confidence Scores')
    plt.legend()
    ### save the histogram    
    plt.savefig("./results/uncertainty_histogram.png")
    plt.close()

    ### draw the ROC curve for OOD detection
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve([1] * len(in_scores) + [0] * len(out_scores), np.concatenate([in_scores, out_scores]))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for OOD Detection')
    plt.legend(loc="lower right")
    ### save the ROC curve
    plt.savefig("./results/ood_roc_curve.png")
    plt.close()
    
    return metrics


if __name__ == '__main__':
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size)

    ### train classification model
    # block = Bottleneck
    # num_blocks = [3, 4, 6, 3]
    # classify_model = ResNet(block, num_blocks,input_channels=input_channels, num_classes=num_classes).to(device)
    # print(f"Loading trained classification model from {classification_model_path}")
    # classify_model.load_state_dict(torch.load(classification_model_path))

     ### train eaae model
    if uncertainty_model_name == 'eaFNO':
        uncertainty_model = EABlockFNO(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaCNN':
        uncertainty_model = EABlockCNN(im_x, im_y, hidden_dim, epi_channels, input_channels, modes1, modes2)
    elif uncertainty_model_name == 'eaae':
        uncertainty_model = EAAE(batch_size,device,im_x, im_y, hidden_dim, latent_dim, input_channels,num_classes, modes1, modes2)
    elif uncertainty_model_name == 'compound':
        uncertainty_model = CompoundModel(im_x, im_y, hidden_dim, num_classes, input_channels, modes1, modes2)
    
    print(f"Loading trained uncertainty model from {uncertainty_model_path}")
    uncertainty_model.load_state_dict(torch.load(uncertainty_model_path))
    uncertainty_model = uncertainty_model.to(device)
    in_loader = train_loader  # Use the training set as in-distribution data
    out_loader = valid_loader  # Use the validation set (SVHN) as out-of-distribution data
    # Evaluate OOD detection performance

    # classification_data = np.load(classification_train_outputs_path)
    # all_classification_outputs = classification_data['probs']
    # all_classification_labels = classification_data['labels']
    # all_classification_data = classification_data['data']
    # ## create a new dataloader for uncertainty training
    # classification_outputs_tensor = torch.tensor(all_classification_outputs, dtype=torch.float32)
    # classification_labels_tensor = torch.tensor(all_classification_labels, dtype=torch.long)
    # classification_data_tensor = torch.tensor(all_classification_data, dtype=torch.float32)
    # classification_datasets = torch.utils.data.TensorDataset(classification_data_tensor, classification_outputs_tensor,classification_labels_tensor)
    # # classification_datasets = torch.utils.data.Subset(classification_datasets, subset_indices)
    # classification_train_loader = DataLoader(classification_datasets, batch_size=batch_size, shuffle=True)

    metrics = evaluate_ood_detection(uncertainty_model, in_loader, out_loader, device)
    
    print("OOD Detection Metrics:", metrics)
