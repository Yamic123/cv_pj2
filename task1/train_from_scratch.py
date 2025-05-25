import torch
import torch.nn as nn
from torch import optim
from creat_dataloader import load_data
from get_model import get_model
from train import train_model

# Function to train from scratch (for comparison)
def train_from_scratch(model_name="resnet18", batch_size=32, num_epochs=25, 
                      lr=0.01, experiment_name="scratch"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader, num_classes = load_data(batch_size)
    
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    trained_model = train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=num_epochs, model_name=model_name, experiment_name=experiment_name
    )
    
    return trained_model
