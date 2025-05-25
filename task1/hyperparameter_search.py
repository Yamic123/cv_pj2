from creat_dataloader import load_data
from get_model import get_model
from train import train_model
import torch.optim as optim
import torch.nn as nn
import torch
from evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hyperparameter_search(model_name="resnet18"):

    learning_rates = [0.0001, 0.001]
    new_layer_multipliers = [5, 10]
    batch_sizes = [32, 64]
    num_epochs = 15 
    
    results = []
    
    for lr in learning_rates:
        for multiplier in new_layer_multipliers:
            for bs in batch_sizes:
                print(f"Training with lr={lr}, multiplier={multiplier}, batch_size={bs}")
                experiment_name = f"hp_lr{lr}_mult{multiplier}_bs{bs}"
                
                train_loader, test_loader, val_loader, num_classes = load_data(bs)
                
                model = get_model(model_name, pretrained=True, num_classes=num_classes)
                model = model.to(device)
                
                if model_name == "resnet18":
                    new_params = model.fc.parameters()
                    pretrained_params = [p for name, p in model.named_parameters() if "fc" not in name]
                elif model_name == "alexnet":
                    new_params = model.classifier[6].parameters()
                    pretrained_params = [p for name, p in model.named_parameters() if "classifier.6" not in name]
                
                optimizer = optim.SGD([
                    {'params': pretrained_params, 'lr': lr},
                    {'params': new_params, 'lr': lr * multiplier}
                ], momentum=0.9, weight_decay=5e-4)
                

                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                

                criterion = nn.CrossEntropyLoss()
                

                model = train_model(
                    model, train_loader, val_loader, optimizer, criterion, scheduler,
                    num_epochs=num_epochs, model_name=model_name, experiment_name=experiment_name
                )
                

                accuracy = evaluate_model(model, test_loader, model_name, experiment_name)
                
                results.append({
                    'lr': lr,
                    'multiplier': multiplier,
                    'batch_size': bs,
                    'accuracy': accuracy
                })
    
    # Print all results
    print("\nHyperparameter Search Results:")
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"lr={result['lr']}, multiplier={result['multiplier']}, "
              f"batch_size={result['batch_size']}, accuracy={result['accuracy']:.4f}")
    
    # Return best hyperparameters
    best_result = max(results, key=lambda x: x['accuracy'])
    return best_result
