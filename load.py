import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import config


def load_client_data(cid: int, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(os.path.join(data_dir, f'client_{cid}.csv'))

    # Separate features and labels
    X = df.drop(columns=["label"]).values.astype("float32") / 255.0  # normalize
    y = df["label"].values.astype("int64")

    # Reshape X to N x 1 x 28 x 28 (needed for CNNs)
    X = X.reshape(-1, 1, 28, 28)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    # Train/val split depending on how many classes there are
    if len(unique_classes) == 1 or min_class_count == 1:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=None, random_state=config.SEED)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.SEED)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader