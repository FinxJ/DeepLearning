import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

#transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128)
])


data_dir='ants_bees'
image_datasets={}
image_datasets['train']=datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
image_datasets['val']=datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

# Data loaders
train_loader = DataLoader(image_datasets['train'], batch_size=128, shuffle=True)
val_loader = DataLoader(image_datasets['val'], batch_size=128, shuffle=False)

#load pretrained model
model=models.resnet18(pretrained=True)
print(model)

#freeze all the layers initially
for param in model.parameters():
    param.requires_grad = False

#replace the final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: ants and bees

criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training
epoch_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(10):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Forward pass
        output = model(images)
        loss = criterian(output, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate training metrics
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    epoch_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/10] - Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.2f}% - Val Acc: {val_accuracy:.2f}%')

print(f'\n✅ Final Training Accuracy: {train_accuracies[-1]:.2f}%')
print(f'✅ Final Validation Accuracy: {val_accuracies[-1]:.2f}%')
