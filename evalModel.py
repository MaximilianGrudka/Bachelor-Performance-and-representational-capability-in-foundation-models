import os
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
from timm import create_model

# Custom Dataset for Camelyon17
class Camelyon17Dataset(Dataset):
    def __init__(self, metadata_csv, image_dir, split=1, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        patient = row["patient"]
        node = row["node"]
        x = row["x_coord"]
        y = row["y_coord"]
        label = int(row["tumor"])

        img_name = f"patient_{patient:03d}_node_{node}/patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Evaluation


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_embeddings:
        # Load .npz embeddings
        data = np.load(args.embedding_npz)
        X = torch.tensor(data['images'], dtype=torch.float32)
        y = torch.tensor(data['labels'], dtype=torch.long)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Create linear head and load weights
        input_dim = X.shape[1]
        model = nn.Sequential(nn.Linear(input_dim, args.num_classes))
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded linear head from {args.checkpoint}")

    else:
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        if args.dataset.lower() == "camelyon":
            dataset = Camelyon17Dataset(
                metadata_csv=os.path.join(args.image_dir, "metadata.csv"),
                image_dir=os.path.join(args.image_dir, "patches"),
                split=1,
                transform=transform
            )
        else:
            dataset = datasets.ImageFolder(args.image_dir, transform=transform)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = create_model(args.model_name, pretrained=not args.checkpoint, num_classes=args.num_classes)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded model from checkpoint: {args.checkpoint}")

    model.to(device)
    print("Evaluating...")
    evaluate_model(model, dataloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., imagenet or camelyon)")
    parser.add_argument("--image_dir", type=str, help="Path to image directory (ImageFolder style)")
    parser.add_argument("--embedding_npz", type=str, help="Path to .npz file with precomputed embeddings")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--model_name", type=str, help="Timm model name (if using full model)")
    parser.add_argument("--use_embeddings", action="store_true", help="Use .npz and linear head")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()
    main(args)
