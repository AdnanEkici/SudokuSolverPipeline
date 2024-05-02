import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os
from sudoku_digit_dataset import SudokuDigitDataset
from tqdm import tqdm

# Define your neural network model
class Dnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, padding=1, kernel_size=(3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Initialize the dataset
train_data_path = r'C:\Users\Pc\Desktop\Sudoku-Dataset\extracted_digit_data\train'
train_dataset = SudokuDigitDataset(root_dir=train_data_path, transform=transform)

# Initialize the DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Initialize your model
model = Dnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
# Train the model
for epoch in tqdm(range(100)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass + backward pass + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
torch.save(model.state_dict(), 'model.pth')
print('Finished Training')