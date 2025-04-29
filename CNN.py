import os
import time
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "./Data"
OUTPUT_DIR  = "./results"
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0    # safe on Windows
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── UTILITIES ──────────────────────────────────────────────────────────────────
def remove_corrupted(dataset):
    good = []
    for path, label in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, label))
        except Exception:
            print(f"Skipping corrupted image: {path}")
    dataset.samples = good
    dataset.targets = [lbl for _, lbl in good]

def infer_image_size(train_dir):
    # pick first image to get its size
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg","bmp","gif")):
                with Image.open(os.path.join(root,f)) as img:
                    return img.size  # (W, H)
    raise RuntimeError("No images found in TRAIN")

class CustomCNN(nn.Module):
    def __init__(self, in_channels, input_size, num_classes):
        super().__init__()
        W, H = input_size
        # ensure divisibility by 8
        W8, H8 = W//8, H//8
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * H8 * W8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    print("Using device:", DEVICE)
    for dataset in os.listdir(DATA_ROOT):
        ds_path = os.path.join(DATA_ROOT, dataset)
        if not os.path.isdir(ds_path):
            continue

        out_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset}_cnn.csv")
        if os.path.exists(out_csv):
            print(f"→ Skipping {dataset}: results already exist.")
            continue

        print(f"\n>>> Dataset: {dataset}")

        # Paths
        train_dir = os.path.join(ds_path, "TRAIN")
        test_dir  = os.path.join(ds_path, "TEST")

        # infer dataset‐specific image size
        W, H = infer_image_size(train_dir)
        print(f"  Inferred image size: {W}×{H}")

        # Transforms
        transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # Datasets
        train_ds = datasets.ImageFolder(train_dir, transform=transform)
        test_ds  = datasets.ImageFolder(test_dir,  transform=transform)
        remove_corrupted(train_ds)
        remove_corrupted(test_ds)

        class_names = train_ds.classes
        print(f"  TRAIN: {len(train_ds)} images; TEST: {len(test_ds)} images; classes: {class_names}")

        # Loaders
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        # Model
        cnn = CustomCNN(
            in_channels=3,
            input_size=(W, H),
            num_classes=len(class_names)
        ).to(DEVICE)
        print("Model parameters:", sum(p.numel() for p in cnn.parameters()))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=LR)

        # Train
        cnn.train()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            t0 = time.time()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = cnn(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * imgs.size(0)
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} — loss {epoch_loss/len(train_ds):.4f} — {time.time()-t0:.1f}s")

        # Evaluate
        cnn.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = cnn(imgs)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())

        y_true_names = [class_names[i] for i in y_true]
        y_pred_names = [class_names[i] for i in y_pred]

        # Metrics
        rpt = classification_report(
            y_true_names, y_pred_names,
            labels=class_names, output_dict=True, zero_division=0
        )
        df = pd.DataFrame(rpt).T
        df.loc["accuracy", :] = [
            accuracy_score(y_true_names, y_pred_names),
            None, None, len(y_true_names)
        ]

        kappa = cohen_kappa_score(y_true_names, y_pred_names)
        cm    = confusion_matrix(y_true_names, y_pred_names, labels=class_names)
        n     = len(y_true_names)
        tp, fp, fn, tn = {}, {}, {}, {}
        for i, cls in enumerate(class_names):
            tp[cls] = int(cm[i,i])
            fp[cls] = int(cm[:,i].sum() - cm[i,i])
            fn[cls] = int(cm[i,:].sum() - cm[i,i])
            tn[cls] = n - tp[cls] - fp[cls] - fn[cls]
        df["tp"], df["fp"], df["fn"], df["tn"] = (
            pd.Series(tp), pd.Series(fp), pd.Series(fn), pd.Series(tn)
        )
        df.loc["cohen_kappa", ["precision","recall","f1-score","support","tp","fp","fn","tn"]] = (
            [None, None, kappa, None, None, None, None, None]
        )

        df.to_csv(out_csv, index=True)
        print(f"→ Saved metrics to {out_csv}")

if __name__ == "__main__":
    freeze_support()
    main()
