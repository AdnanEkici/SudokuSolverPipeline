import torch
from torchvision import transforms
from sudoku_digit_dataset import SudokuDigitDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
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
import cv2

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

# Define the transformation for the test dataset (similar to the training and validation datasets)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Initialize the test dataset
test_data_path = "d_net_dataset/validation"
test_dataset = SudokuDigitDataset(root_dir=test_data_path, transform=transform)

# Initialize the DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # No need to shuffle for inference

# Initialize the model
model = Dnet()
model.load_state_dict(torch.load('model.pth'))  # Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define a function to perform inference on the test set and visualize predictions
def inference_with_visualization(model, test_loader, device):
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            # Visualize predictions
            for i in range(inputs.size(0)):
                image = inputs[i].cpu().numpy().squeeze()
                # Rescale image values from [0, 1] to [0, 255] and convert to uint8
                image = (image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for OpenCV
                image = cv2.resize(image, (600 , 600))
                cv2.putText(image, 'Label: {}'.format(labels[i].item()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, 'Prediction: {}'.format(predicted[i].item()), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Prediction', image)
                cv2.waitKey(0)  # Wait for a key press to continue
                cv2.destroyAllWindows()
    return predictions

# Call the function
test_predictions = inference_with_visualization(model, test_loader, device)