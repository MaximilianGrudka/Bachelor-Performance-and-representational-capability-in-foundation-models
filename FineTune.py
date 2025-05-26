import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from timm.data import create_transform

def train(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    """
    Fine-tune a model's classifier head using linear probing on ImageNet-1K.
    """
    start_time = time.time()
    print("\tStart fine-tuning (Linear Probing) ...")

    # Load model from timm
    print("\tLoading model...")
    model = timm.create_model(config['architecture'], pretrained=True)

    # Replace classifier head for linear probing
    num_features = model.num_features
    model.head = nn.Linear(num_features, config['num_classes'])

    # Freeze all layers except classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    model = model.to(config['device'])

    # Define optimizer and scheduler
    optimizer = AdamW(model.head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("max_epochs", 100))

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(config['device'])

    best_loss, best_epoch = np.inf, 0
    best_model = deepcopy(model)
    epochs_no_improve = 0
    n_epochs_stop = config.get("early_stop_patience", 10)
    max_epochs = config.get("max_epochs", 100)

    # Training loop with mixed precision
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

        # Validation
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

        # Save checkpoints
        checkpoint_dir = Path(config['output_path'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        architecture_name = config['architecture'].replace(':', '_').replace('/', '_')
        checkpoint_path = checkpoint_dir / f"{architecture_name}_epoch{epoch+1}_checkpoint.pth"
        torch.save(model.state_dict(), str(checkpoint_path))
        print(f"\tCheckpoint saved: {checkpoint_path}")

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_epochs_stop:
            print("\tEarly stopping triggered!")
            break

    best_model_dir = Path(config['output_path'])
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / f"{config.get('dataset', 'imagenet1k')}_{architecture_name}_best.pth"
    torch.save(best_model.state_dict(), str(best_model_path))
    print(f"\tBest model saved: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    # Use DINO ImageNet preprocessing
    transform = create_transform(
        input_size=224,
        is_training=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_dataset = ImageFolder(root=f"{config['data_path']}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{config['data_path']}/val", transform=transform)
    config['num_classes'] = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=8, pin_memory=True)

    train(config, train_loader, val_loader)
