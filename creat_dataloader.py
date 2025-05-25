import os
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


dataset_path = "./caltech-101/101_ObjectCategories" 


class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != 'BACKGROUND_Google']
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            
            files.sort()
            _files = files[:int(len(files)*0.8)]
            train_files = _files[:int(len(_files)*0.8)]
            val_files = _files[int(len(_files)*0.8):]
            test_files = files[int(len(files)*0.8):]
            
            if mode == 'train':
                for f in train_files:
                    self.images.append(Image.open(os.path.join(class_dir, f)).convert('RGB'))
                    self.labels.append(class_idx)
            elif mode == 'val':
                for f in val_files:
                    self.images.append(Image.open(os.path.join(class_dir, f)).convert('RGB'))
                    self.labels.append(class_idx)
            elif mode == 'test':
                for f in test_files:
                    self.images.append(Image.open(os.path.join(class_dir, f)).convert('RGB'))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data(batch_size=32):
    train_dataset = Caltech101Dataset(dataset_path, transform=data_transforms['train'], mode='train')
    test_dataset = Caltech101Dataset(dataset_path, transform=data_transforms['val'], mode='test')
    val_dataset = Caltech101Dataset(dataset_path, transform=data_transforms['val'], mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    return train_loader, test_loader,val_loader, num_classes
