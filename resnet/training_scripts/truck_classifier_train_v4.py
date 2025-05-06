import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # for progress bar
import yaml  # for output yaml file

# --- CONFIGURATIONS --- #
run_number = 1  # Increment for each run if needed
batchSize = 2  # originally 16 but we'll go up in powers of 2 and test it out
learning_rate = 1e-4
num_epochs = 15
modelName = 'resnet18'
model_weight_fname = f'best_truck_classifier{run_number}.pth'
# ---------------------- #

def save_hyperparameters_to_yaml(run_number, batch_size, num_epochs, optimizer, lr, model_name, output_dir='./runs'):
    run_dir = os.path.join(output_dir, f'run{run_number}')
    os.makedirs(run_dir, exist_ok=True)
    config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'optimizer': optimizer.__class__.__name__,
        'learning_rate': lr,
        'model': model_name
    }
    yaml_path = os.path.join(run_dir, 'hyperparameters.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)
    print(f'Hyperparameters saved to {yaml_path}')

# Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data augmentation and transformations for training and validation
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Grayscale if needed
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),  # Vertical flip with a small chance
    transforms.RandomRotation(15),  # Rotate images by a random degree between -15 and 15
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

# Load the datasets
train_data = datasets.ImageFolder(
    'C:/Users/btros/Documents/GitHub/CARL/data/resnet_data/train', 
    transform=transform_train
)
val_data = datasets.ImageFolder(
    'C:/Users/btros/Documents/GitHub/CARL/data/resnet_data/val', 
    transform=transform_val
)

train_loader = DataLoader(
    train_data, 
    batch_size=batchSize, 
    shuffle=True
    )
val_loader = DataLoader(
    val_data, 
    batch_size=batchSize, 
    shuffle=False
    )

# Load a pre-trained ResNet model and modify it
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: truck and non-truck
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoint directory
checkpoint_dir = '../checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
# Runs directory
runs_dir = '../runs'
os.makedirs(runs_dir, exist_ok=True)

# Training loop
best_val_accuracy = 0.0
final_model_path = f'../runs/run{run_number}/{model_weight_fname}'

save_hyperparameters_to_yaml(
    run_number=run_number,
    batch_size=batchSize,
    num_epochs=num_epochs,
    optimizer=optimizer,
    lr=learning_rate,  # Your learning rate
    model_name=modelName,  # Your model name
    output_dir=runs_dir  # Path to the folder where runs are stored
)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for training
    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
    
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

    # Validate the model
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Loss: {val_running_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Checkpointing
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
            'val_accuracy': val_accuracy,
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

        torch.save(model.state_dict(), final_model_path)
        print(f'Better Val Accuracy best model saved/updated: {final_model_path}')
