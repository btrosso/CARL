import os
from PIL import Image
from torch.utils.data import Dataset

class VehicleDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels  # This can be a dict or a list, we’ll define format later
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pass  # We'll fill this in next
