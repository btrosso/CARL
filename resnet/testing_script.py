import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformation for test data
transform_test = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),  # Keep if you used grayscale in training
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

# Load the test dataset
test_data = datasets.ImageFolder(
    'C:/Users/btros/Documents/GitHub/CARL/data/resnet_data/test', 
    transform=transform_test
    )
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: truck and non-truck
model.load_state_dict(torch.load('C:/Users/btros/Documents/GitHub/CARL/resnet/best_truck_classifier3.pth'))  # Load saved model
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Test the model
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# Calculate accuracy
test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')
