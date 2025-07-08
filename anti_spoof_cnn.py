import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# === 1. Settings ===
data_dir = 'anti_spoof_dataset-1'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'valid')

img_size = 224
batch_size = 32
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 2. Transforms ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === 3. Load Datasets ===
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === 4. Model ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real / spoof
model = model.to(device)

# === 5. Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 6. Training Loop ===
for epoch in range(epochs):
    model.train()
    train_loss, correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(train_dataset)
    print(f"📘 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Accuracy: {accuracy:.4f}")

# === 7. Save the model ===
torch.save(model.state_dict(), "anti_spoofing_model.pth")
print(" Model trained and saved as anti_spoofing_model.pth")

# === 8. Evaluation on Validation Set ===
def evaluate(model, loader, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n Classification Report (Validation Set):")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate(model, val_loader, train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)