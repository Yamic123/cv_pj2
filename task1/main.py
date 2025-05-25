from train_different_lr import train_with_different_lr
from train_from_scratch import train_from_scratch
from evaluate import evaluate_model
from hyperparameter_search import hyperparameter_search
from creat_dataloader import load_data

def main():
    print("Starting Caltech-101 fine-tuning experiment")
    
    # print("\n--- Hyperparameter Search ---")
    # best_params = hyperparameter_search(model_name="alexnet")
    # print(f"Best hyperparameters: {best_params}")
    
    best_params = {
        'lr': 0.001,
        'multiplier': 10,
        'batch_size': 64
    }

    print("\n--- Training with Pre-trained Weights ---")
    pretrained_model = train_with_different_lr(
        model_name="resnet18",
        batch_size=best_params['batch_size'],
        num_epochs=30,  
        base_lr=best_params['lr'],
        new_layer_lr_multiplier=best_params['multiplier'],
        experiment_name="pretrained_best"
    )
    

    print("\n--- Training from Scratch ---")
    scratch_model = train_from_scratch(
        model_name="resnet18",
        batch_size=32,
        num_epochs=30,  
        lr=0.001 * 10,  
        experiment_name="scratch_best"
    )
    

    _, test_loader, _,_ = load_data(batch_size=best_params['batch_size'])
    

    print("\n--- Final Evaluation ---")
    pretrained_acc = evaluate_model(pretrained_model, test_loader, "resnet18", "pretrained_best")
    scratch_acc = evaluate_model(scratch_model, test_loader, "resnet18", "scratch_best")
    
    print(f"\nPre-trained model accuracy: {pretrained_acc:.4f}")
    print(f"From-scratch model accuracy: {scratch_acc:.4f}")
    print(f"Improvement from pre-training: {pretrained_acc - scratch_acc:.4f}")

if __name__ == "__main__":
    main()
