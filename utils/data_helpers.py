from torchvision import datasets
import torch 
from torchvision.transforms import v2
from torch.utils.data import DataLoader

train_data = datasets.CIFAR10(
    root = 'data',
    train = True,                         
    transform = v2.ToTensor(), 
    download = True,            
)
test_data = datasets.CIFAR10(
    root = 'data', 
    train = False, 
    transform = v2.ToTensor()
)

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
