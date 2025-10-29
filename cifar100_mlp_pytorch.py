import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.CIFAR100(root='./dir', train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root='./dir', train=False, download=True, transform=transform)

# Data loader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

#Architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 100)
    
    def forward(self, x):
        x=self.flatten(x)
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=torch.relu(x)
        x=self.fc3(x)
        x=torch.relu(x)
        x=self.fc4(x)
        x=torch.relu(x)
        x=self.fc5(x)
        return x
    
model = MLP()
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training
epoch_loss=[]
for epoch in range(10):
    for images, labels in train_loader:
        output = model(images)
        loss = criterian(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss.append(loss.item())
    print(f"epoch_loss:{epoch_loss}")

#Testing
total=0.0
correct=0.0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        output =model(images)
        __, predicted = torch.max(output, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
accuracy = correct/total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
