
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, 
                num_epochs=25, model_name="resnet18", experiment_name="finetune"):
    
    writer = SummaryWriter(log_dir=f'cv_pj2/runs/{model_name}_{experiment_name}')
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }
    best_acc = 0.0
    wait = 0
    
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            progress_bar.set_postfix(loss=loss.item(), 
                                    acc=torch.sum(preds == labels.data).item() / inputs.size(0))
        
        if scheduler:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        results["train_loss"].append(epoch_loss)
        results["train_acc"].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
        
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=loss.item(), 
                                        acc=torch.sum(preds == labels.data).item() / inputs.size(0))
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        results["val_loss"].append(val_epoch_loss)
        results["val_acc"].append(val_epoch_acc.item())

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": epoch_loss,
                                            "val_loss": val_epoch_loss},
                           global_step=epoch)

        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": epoch_acc,
                                            "val_acc": val_epoch_acc}, 
                           global_step=epoch)
        
        improvement = (val_epoch_acc - best_acc).item()
        epsilon = 1e-5
        if improvement > 0.001 + epsilon:
            print("Saving best model")
            best_acc = val_epoch_acc
            wait = 0
            save_path = Path(f'cv_pj2/models/{model_name}_{experiment_name}_best.pth')
            torch.save(model.state_dict(), save_path)
        else:
            print("No improvement")
            wait += 1
            if wait >= 5:
                print("Early stopping")
                break
        print(wait,improvement,best_acc)
    writer.close()
    print(f'Best val Acc: {best_acc:.4f}')
    return model
