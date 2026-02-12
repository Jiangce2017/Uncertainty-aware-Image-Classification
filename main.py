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

from model import get_model, MODELS

# Default hyperparameters
lr = 1e-4
log_interval = 10
epochs = 200
batch_size = 64
data_root = '/ocean/projects/cis240139p/jchen39/datasets' 

class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def train(model, device, train_loader, epoch, verbose=True):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    total_class_loss = 0
    total_epi_error = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        class_loss = F.cross_entropy(output, target)
        loss = class_loss 
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_class_loss += class_loss.item()
        
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    accuracy = 100. * correct / len(train_loader.dataset)
    total_class_loss /= (batch_idx + 1)
    
    return total_class_loss,accuracy, total_epi_error

def two_stage_train(train_stage, model, device, train_loader, epoch, verbose=True):
    model.train()
    if train_stage ==1:
        print("Stage one: training only classifier blocks")
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.ea_block.parameters():
            param.requires_grad = False
    elif train_stage ==2:
        print("Stage two: training only EA blocks")
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.ea_block.parameters():
            param.requires_grad = True
    ## set optimizer to make sure only classifier parameters are updated
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_class_loss = 0
    total_epi_error = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, epi_error = model(data)
        class_loss = F.cross_entropy(output, target)
        var_loss = torch.mean(epi_error)
        if train_stage ==1:
            loss = class_loss
        elif train_stage ==2:
            loss = var_loss
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_class_loss += class_loss.item()
        total_epi_error += var_loss.item()
        
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    accuracy = 100. * correct / len(train_loader.dataset)
    total_class_loss /= (batch_idx + 1)
    total_epi_error /= (batch_idx + 1)
    
    return total_class_loss,accuracy, total_epi_error
    
def test(model, device, test_loader, verbose=True):
    model.eval()
    total_class_loss = 0
    correct = 0
    total_epi_error = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, epi_error = model(data)
            total_class_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_epi_error += torch.mean(epi_error).item()
    
    total_class_loss /= (batch_idx + 1)
    accuracy = 100. * correct / len(test_loader.dataset)
    total_epi_error /= (batch_idx + 1)
    
    if verbose:
        print(f'\nTest set: Average loss: {total_class_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({accuracy:.0f}%)\n')
    
    return total_class_loss, accuracy, total_epi_error

def load_dataset(dataset_name, data_root, batch_size):
    """Load dataset and return DataLoader objects for training and testing."""
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=data_root, train=True, download=True,
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=data_root, train=False, transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True,
                                         transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=transforms.ToTensor())
        valid_dataset = datasets.SVHN(root=data_root, split='test', download=True,
                                      transform=transforms.ToTensor())
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True,
                                          transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(root=data_root, train=False, transform=transforms.ToTensor())
        v
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_dataset = torch.utils.data.Subset(train_dataset, range(128))  # Use a subset for faster benchmarking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.Subset(test_dataset, range(128))  # Use a subset for faster benchmarking
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_dataset = torch.utils.data.Subset(valid_dataset, range(128))  # Use a subset for faster benchmarking
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader

def benchmark_model(model_name, device='cpu', epochs=10, lr=1e-3, batch_size=64, ea=False, save_results=False):
    # Load data
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size)

    # Create model
    model = get_model(model_name, input_channels=input_channels, output_channels=output_channels).to(device)
    #optimizer = Adam(model.parameters(), lr=lr)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    
    # Training with fixed number of epochs
    start_time = time.time()
    train_losses = []
    test_losses = []
    test_accuracies = []

    results_dir = Path("/jet/home/jchen39/projects/EAAE/results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if ea:
        experiment_name = model_name+"_"+dataset_name
    else:
        experiment_name = "not_"+model_name+"_"+dataset_name    
    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, experiment_name+'_train.log'),
        ['ep', 'train_class_loss','train_class_acc','train_epi_error']
    )
    test_logger = Logger(
        osp.join(results_dir, experiment_name+'_test.log'),
        ['ep', 'test_class_loss','test_class_acc','test_epi_error']
    )
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_class_loss, train_class_acc, train_epi_error = train(model, device, train_loader, epoch, verbose=False)
        test_class_loss, test_class_acc, test_epi_error = test(model, device, test_loader, verbose=False)
        train_losses.append(train_class_loss)
        test_losses.append(test_class_loss)
        test_accuracies.append(test_class_acc)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{epochs}: Train Loss: {train_class_loss:.4f}, Train Accuracy: {train_class_acc:.2f}%, Train Epi Error: {train_epi_error:.6f}, Test Loss: {test_class_loss:.4f}, Test Acc: {test_class_acc:.2f}%, Test Epi Error: {test_epi_error:.6f}, Time: {epoch_time:.1f}s")
        # Log epoch results
        train_logger.log({
            'ep': epoch,
            'train_class_loss': train_class_loss,
            'train_class_acc': train_class_acc,
            'train_epi_error': train_epi_error,
        })
        test_logger.log({
            'ep': epoch,
            'test_class_loss': test_class_loss,
            'test_class_acc': test_class_acc,
            'test_epi_error': test_epi_error,
        })
    total_time = time.time() - start_time
        
    # Final results
    final_test_acc = test_accuracies[-1]
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Training Time: {total_time:.2f}s")
    print(f"Time per Epoch: {total_time/(2*epochs):.2f}s")
        
    # Save model
    models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, dataset_name+"_"+model_name+".pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    

def ea_model(model_name, device='cpu', epochs=10, lr=1e-3, batch_size=64, ea=False, save_results=False):
    """Benchmark a single model with fixed number of epochs"""
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name.upper()}")
    print(f"{'='*50}")

    # Load data
    train_loader, test_loader, valid_loader = load_dataset(dataset_name, data_root, batch_size)
    # Create model
    model = get_model(model_name, input_channels=input_channels, output_channels=output_channels, im_x=im_x, im_y=im_y, hidden_dim=hidden_dim, epi_channels=epi_channels, ea=ea).to(device)
    #optimizer = Adam(model.parameters(), lr=lr)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    
    # Training with fixed number of epochs
    start_time = time.time()
    train_losses = []
    test_losses = []
    test_accuracies = []

    results_dir = Path("/jet/home/jchen39/projects/EAAE/results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if ea:
        experiment_name = model_name+"_"+dataset_name
    else:
        experiment_name = "not_"+model_name+"_"+dataset_name
        
    ## setup logger
    train_logger = Logger(
        osp.join(results_dir, experiment_name+'_train.log'),
        ['ep', 'train_class_loss','train_class_acc','train_epi_error','stage']
    )
    test_logger = Logger(
        osp.join(results_dir, experiment_name+'_test.log'),
        ['ep', 'test_class_loss','test_class_acc','test_epi_error','stage']
    )

    ### setup a two-stage training, the first stage, we train only classifier blocks, in the second stage, we only train the EA blocks
    print("Starting training...")
    for train_stage in [1,2]:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_class_loss, train_class_acc, train_epi_error = two_stage_train(train_stage, model, device, train_loader, epoch, verbose=False)
            test_class_loss, test_class_acc, test_epi_error = test(model, device, test_loader, verbose=False)
            valid_class_loss, valid_class_acc, valid_epi_error = test(model, device, valid_loader, verbose=False)
            train_losses.append(train_class_loss)
            test_losses.append(test_class_loss)
            test_accuracies.append(test_class_acc)
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch:2d}/{epochs}: Train Loss: {train_class_loss:.4f}, Train Accuracy: {train_class_acc:.2f}%, Train Epi Error: {train_epi_error:.6f}, Test Loss: {test_class_loss:.4f}, Test Acc: {test_class_acc:.2f}%, Test Epi Error: {test_epi_error:.6f}, Time: {epoch_time:.1f}s")
            print(f"Valid Loss: {valid_class_loss:.4f}, Valid Acc: {valid_class_acc:.2f}%, Valid Epi Error: {valid_epi_error:.6f}")
            # Log epoch results
            train_logger.log({
                'ep': epoch,
                'train_class_loss': train_class_loss,
                'train_class_acc': train_class_acc,
                'train_epi_error': train_epi_error,
                'stage': train_stage
            })
            test_logger.log({
                'ep': epoch,
                'test_class_loss': test_class_loss,
                'test_class_acc': test_class_acc,
                'test_epi_error': test_epi_error,
                'stage': train_stage 
            })
    total_time = time.time() - start_time   
    # Final results
    final_test_acc = test_accuracies[-1]
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Training Time: {total_time:.2f}s")
    print(f"Time per Epoch: {total_time/epochs:.2f}s")
        
    # Save model
    models_dir = Path("/jet/home/jchen39/projects/EAAE/checkpoints")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, dataset_name+"_"+model_name+".pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'  # Device to use: 'cpu' or 'cuda'
# Note: epochs is already defined above in the default hyperparameters section

### Specify dataset
dataset_name = 'cifar10' #'mnist'  
if dataset_name == 'mnist':
    im_x=28
    im_y=28
    input_channels = 1
    output_channels = 10
    epi_channels=10
elif dataset_name == 'cifar10':
    im_x=32
    im_y=32
    input_channels = 3
    output_channels = 10
    epi_channels=10
elif dataset_name == 'cifar100':
    im_x=32
    im_y=32
    input_channels = 3
    output_channels = 100
    epi_channels=100
hidden_dim=64
save_results=True

def main():
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        actual_device = 'cpu'
    else:
        actual_device = device
    model = 'earesnet50'  # earesnet50 or eavgg, or mcdropout_resnet50
    ea = True
    if ea:
        ea_model(model, actual_device, epochs, lr, batch_size, ea=ea, save_results=save_results)
    else:
        benchmark_model(model, actual_device, epochs, lr, batch_size, ea=ea, save_results=save_results) 

if __name__ == '__main__':
    main()