import torch
import torch.nn as nn
import torch.optim as optim
from creat_dataloader import load_data
from get_model import get_model
from train import train_model
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_different_lr(model_name="resnet18", batch_size=32, num_epochs=25,
                           base_lr=0.01, new_layer_lr_multiplier=10, experiment_name="finetune"):
    
    train_loader, test_loader, val_loader, num_classes = load_data(batch_size)
    
    model = get_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    if model_name == "resnet18":
        new_params = model.fc.parameters()
        pretrained_params = [p for name, p in model.named_parameters() if "fc" not in name]
    elif model_name == "alexnet":
        new_params = model.classifier[6].parameters()
        pretrained_params = [p for name, p in model.named_parameters() if "classifier.6" not in name]
    
    optimizer = optim.SGD([
        {'params': pretrained_params, 'lr': base_lr},
        {'params': new_params, 'lr': base_lr * new_layer_lr_multiplier}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    trained_model = train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=num_epochs, model_name=model_name, experiment_name=experiment_name
    )
    
    return trained_model
