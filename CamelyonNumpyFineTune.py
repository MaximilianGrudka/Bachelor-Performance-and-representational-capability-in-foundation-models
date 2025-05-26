import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from timm.data import create_transform
from typing import Optional, Callable
from wilds import get_dataset

class NumpyDataset(Dataset):
    def __init__(self, npz_file_path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.data = np.load(npz_file_path)
        self.images = self.data['images']
        self.targets = self.data['labels']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def extract_embeddings(model, device, dataloader):
    embeddings_db, labels_db = [], []
    model.eval()
    with torch.no_grad():
        for images, labels, *_ in tqdm(dataloader):
            images = images.to(device)
            output = model.forward_features(images)
            output = model.forward_head(output, pre_logits=True)
            output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    return {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }


def train(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    print("\tStart fine-tuning (Linear Probing) ...")

    if config.get('use_embeddings', False):
        input_dim = train_loader.dataset.images.shape[1]
        model = nn.Sequential(nn.Linear(input_dim, config['num_classes']))
    else:
        print("\tLoading model...")
        model = timm.create_model(config['architecture'], pretrained=True)
        num_features = model.num_features
        model.head = nn.Linear(num_features, config['num_classes'])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    model = model.to(config['device'])

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("max_epochs", 100))
    criterion = nn.CrossEntropyLoss().to(config['device'])

    best_loss = float('inf')
    best_model = deepcopy(model)
    epochs_no_improve = 0
    n_epochs_stop = config.get("early_stop_patience", 10)
    max_epochs = config.get("max_epochs", 100)
    scaler = torch.amp.GradScaler()

    for epoch in range(max_epochs):
        print(f"\tEpoch {epoch + 1}/{max_epochs}")
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=config['device']):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        checkpoint_dir = Path(config['output_path'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        architecture_name = config['architecture'].replace(':', '_').replace('/', '_')
        checkpoint_path = checkpoint_dir / f"{architecture_name}_epoch{epoch+1}_checkpoint.pth"
        torch.save(model.state_dict(), str(checkpoint_path))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_epochs_stop:
            print("\tEarly stopping triggered!")
            break

    best_model_path = checkpoint_dir / f"{config.get('dataset', 'wilds')}_{architecture_name}_best.pth"
    torch.save(best_model.state_dict(), str(best_model_path))
    print(f"\tBest model saved: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    transform = create_transform(input_size=224, is_training=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    dataset = get_dataset("camelyon17", root_dir=config['data_path'], download=False)

    if config.get('use_embeddings', False):
        print("Using precomputed embeddings for linear probing.")

        train_npz = output_path / 'train_embeddings.npz'
        val_npz = output_path / 'val_embeddings.npz'

        if not train_npz.exists() or not val_npz.exists():
            print("Extracting embeddings...")
            model = timm.create_model(config['architecture'], pretrained=True)
            model = model.to(config['device'])
            model.eval()

            train_dataset_raw = dataset.get_subset("train", transform=transform)
            val_dataset_raw = dataset.get_subset("val", transform=transform)

            train_loader_raw = DataLoader(train_dataset_raw, batch_size=config['batch_size'], shuffle=False, num_workers=8)
            val_loader_raw = DataLoader(val_dataset_raw, batch_size=config['batch_size_eval'], shuffle=False, num_workers=8)

            train_embed = extract_embeddings(model, config['device'], train_loader_raw)
            val_embed = extract_embeddings(model, config['device'], val_loader_raw)

            np.savez(train_npz, images=train_embed['embeddings'], labels=train_embed['labels'])
            np.savez(val_npz, images=val_embed['embeddings'], labels=val_embed['labels'])

        train_dataset = NumpyDataset(str(train_npz))
        val_dataset = NumpyDataset(str(val_npz))
        config['num_classes'] = len(np.unique(train_dataset.targets))

    else:
        train_dataset = dataset.get_subset("train", transform=transform)
        val_dataset = dataset.get_subset("val", transform=transform)
        config['num_classes'] = dataset.n_classes

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=8, pin_memory=True)

    train(config, train_loader, val_loader)
