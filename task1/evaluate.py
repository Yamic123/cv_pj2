import torch
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, model_name="resnet18", experiment_name="finetune"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"{model_name} - {experiment_name} Accuracy: {accuracy:.4f}")
    
    return accuracy
