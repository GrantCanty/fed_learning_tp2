import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import config


def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = config.DATASET_NAME
    
    # Dataset mapping
    dataset_classes = {
        "FashionMNIST": datasets.FashionMNIST,
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataset_class = dataset_classes[dataset_name]
    
    # Different transforms for different datasets
    if dataset_name in ["FashionMNIST", "MNIST"]:
        transform = transforms.ToTensor()
    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.ToTensor()

    train_data = dataset_class(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    targets = np.array(train_data.targets)
    data = np.array(train_data.data)

    num_classes = len(np.unique(targets))
    class_indices = [np.where(targets == y)[0] for y in range(num_classes)]

    client_indices = [[] for _ in range(k)]
    for c in range(num_classes):
        class_idx = class_indices[c]
        np.random.seed(config.SEED)
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, k))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        split_indices = np.split(class_idx, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx.tolist())

    for i, indices in enumerate(client_indices):
        client_data = data[indices]
        client_targets = targets[indices]
        
        flat_images = client_data.reshape(len(client_data), -1)
        df = pd.DataFrame(flat_images)
        df['label'] = client_targets
        
        df.to_csv(os.path.join(save_dir, f'client_{i}.csv'), index=False)

    print(f"Distributed datasets saved to {save_dir}")