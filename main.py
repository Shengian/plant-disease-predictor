import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import ImageFile

# Handle corrupted or large images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_path = "D:/ml training/new linked"  # update if needed
epochs = 5
batch_size = 8  # Reduced from 64 to 8 to avoid MemoryError

# Transformations: Resize images to reduce memory use
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# DataLoader with memory optimizations
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,                # Keep it 0 to reduce memory use
    pin_memory=False,
    persistent_workers=False
)

# Model: Use a simple pretrained model (ResNet18)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Adjust final layer
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} completed with avg loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "trained_model.pth")
print("Training complete. Model saved as 'trained_model.pth'")