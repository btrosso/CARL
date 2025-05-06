import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import shutil
import time

# Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations (same as during training)
transform_inference = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

# Load the pre-trained ResNet model and modify it
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: truck and non-truck
model = model.to(device)

# Load the saved model weights
model_weights_path = 'C:/Users/btros/Documents/GitHub/CARL/resnet/model_weights/'
model.load_state_dict(torch.load(f'{model_weights_path}best_truck_classifier5.pth'))

# Set the model to evaluation mode
model.eval()

# Function to classify an image
def classify_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Apply the transformation
    img = transform_inference(img).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Define source folder and target folders
source_folder = 'C:/Users/btros/Documents/GitHub/CARL/data/stanford_car_dataset/cars_train/cars_train'
truck_folder = 'C:/Users/btros/Documents/GitHub/CARL/data/stanford_car_dataset/cars_train/cars_train/truck'
non_truck_folder = 'C:/Users/btros/Documents/GitHub/CARL/data/stanford_car_dataset/cars_train/cars_train/non_truck'

# Create target folders if they don't exist
os.makedirs(truck_folder, exist_ok=True)
os.makedirs(non_truck_folder, exist_ok=True)

# List all jpg files in the source folder
jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# Loop through the list of jpgs, classify them, and move them
for jpg_file in jpg_files:
    file_path = os.path.join(source_folder, jpg_file)
    
    # Classify the image (0 = non-truck, 1 = truck)
    classification = classify_image(file_path)
    
    # Determine target folder based on classification
    target_folder = truck_folder if classification == 1 else non_truck_folder
    
    # Move the file
    shutil.move(file_path, os.path.join(target_folder, jpg_file))
    
    print(f'Moved {jpg_file} to {"Truck" if classification == 1 else "Non-Truck"} folder.')

print("Classification and moving completed.")
