import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_data(dataset):
    # Load train and test set
    if dataset == "cifar10":
        trainset_full = torchvision.datasets.CIFAR10(root='data', train=True,
                                                download=True, transform=transform)
        testset_full = torchvision.datasets.CIFAR10(root='data', train=False,
                                            download=True, transform=transform)
    elif dataset == "SVHN":
        trainset_full = torchvision.datasets.SVHN(root='data', split='train',
                                                download=True, transform=transform)
        trainset_full.classes = list(range(10))
        testset_full = torchvision.datasets.SVHN(root='data', split='test',
                                            download=True, transform=transform)
        testset_full.classes = list(range(10))
    elif dataset == "cifar100":
        trainset_full = torchvision.datasets.CIFAR100(root='data', train=True,
                                                download=True, transform=transform)
        testset_full = torchvision.datasets.CIFAR100(root='data', train=False,
                                            download=True, transform=transform)
    
    elif dataset == "cifar10_partial":
        trainset_full = torchvision.datasets.CIFAR10(root='data', train=True,
                                                download=True, transform=transform)
        testset_full = torchvision.datasets.CIFAR10(root='data', train=False,
                                            download=True, transform=transform)
        target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        #Filter the training set
        train_indices = [i for i, (_, label) in enumerate(trainset_full) if label in target_classes]
        trainset_filtered = PartialCIFAR10(trainset_full, train_indices)
        # Filter the test set
        test_indices = [i for i, (_, label) in enumerate(testset_full) if label in target_classes]
        testset_filtered = PartialCIFAR10(testset_full, test_indices)
        
        trainset_full = trainset_filtered
        testset_full = testset_filtered
    elif dataset == "SVHN_partial":
        trainset_full = torchvision.datasets.SVHN(root='data', split='train',
                                                download=True, transform=transform)
        trainset_full.classes = list(range(10))
        testset_full = torchvision.datasets.SVHN(root='data', split='test',
                                            download=True, transform=transform)
        testset_full.classes = list(range(10))

        target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        #Filter the training set
        train_indices = [i for i, (_, label) in enumerate(trainset_full) if label in target_classes]
        trainset_filtered = Subset(trainset_full, train_indices)
        trainset_filtered.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Filter the test set
        test_indices = [i for i, (_, label) in enumerate(testset_full) if label in target_classes]
        testset_filtered = Subset(testset_full, test_indices)
        testset_filtered.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        trainset_full = trainset_filtered
        testset_full = testset_filtered
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return trainset_full, testset_full


# Create a custom dataset that wraps the original dataset but only uses selected indices
class PartialCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices
        self.classes = original_dataset.classes

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map the idx to the corresponding index in the original dataset
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]
    

class PartialSVHN(torchvision.datasets.SVHN):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices
        self.classes = original_dataset.classes

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map the idx to the corresponding index in the original dataset
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]