import os
import pandas as pd
import json
import uuid
from datetime import datetime

# Import the training function from SoDaDE_regression
from SoDaDE_regression import train_decoder_once

def run_sodade_regression(dataset="full_data", 
                         freeze_decoder=False,
                         pretrained_model_path="SoDaDE_DM_64_TL_5_heads_16.pth",
                         spange_path="spange_melt.csv",
                         nn_size=16,
                         dropout_fp=0.1,
                         dropout_nn=0.1,
                         epochs=10,
                         hidden_factor=2,
                         val_percentage=0.2,
                         learning_rate_fp=1e-5,
                         learning_rate_nn=1e-4,
                         save_results=True):
    """
    Run SoDaDE regression with preset optimal parameters.
    
    Args:
        dataset (str): Either "single_solvent" or "full_data" (maps to "full_yields" internally)
        freeze_decoder (bool): Whether to freeze the SoDaDE model weights during finetuning
        pretrained_model_path (str): Path to the pretrained SoDaDE model
        spange_path (str): Path to the spange data file
        nn_size (int): Size of the final neural network layer
        dropout_fp (float): Dropout value for the fingerprint model
        dropout_nn (float): Dropout value for the final neural network
        epochs (int): Number of training epochs
        hidden_factor (int): Hidden factor for the model architecture
        val_percentage (float): Validation split percentage
        learning_rate_fp (float): Learning rate for the fingerprint model
        learning_rate_nn (float): Learning rate for the neural network
        save_results (bool): Whether to save results to CSV file
    
    Returns:
        dict: Results dictionary containing MSE scores and parameters
    """
    
    # Map user-friendly dataset names to internal names
    dataset_mapping = {
        "single_solvent": "single_solvent",
        "full_data": "full_yields"
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Dataset must be either 'single_solvent' or 'full_data', got '{dataset}'")
    
    internal_dataset = dataset_mapping[dataset]
    
    print(f"=== Running SoDaDE Regression ===")
    print(f"Dataset: {dataset} (internal: {internal_dataset})")
    print(f"Freeze decoder: {freeze_decoder}")
    print(f"Epochs: {epochs}, NN size: {nn_size}")
    print(f"Dropout FP: {dropout_fp}, Dropout NN: {dropout_nn}")
    print(f"Learning rates - FP: {learning_rate_fp}, NN: {learning_rate_nn}")
    print("=" * 40)
    
    # Call the training function
    results = train_decoder_once(
        dataset=internal_dataset,
        pretrained_model_path=pretrained_model_path,
        spange_path=spange_path,
        freeze_fp=freeze_decoder,
        learning_rate_FP=learning_rate_fp,
        learning_rate_NN=learning_rate_nn,
        dropout_FP=dropout_fp,
        dropout_NN=dropout_nn,
        val_percentage=val_percentage,
        NN_size=nn_size,
        hidden_factor=hidden_factor,
        epochs=epochs
    )
    
    print(f"\n=== Final Results ===")
    print(f"Average MSE: {results['avg_mse']:.4f}")
    
    if save_results:
        save_path = _save_results_to_csv(
            results=results,
            dataset=dataset,
            freeze_decoder=freeze_decoder,
            pretrained_model_path=pretrained_model_path,
            spange_path=spange_path,
            nn_size=nn_size,
            dropout_fp=dropout_fp,
            dropout_nn=dropout_nn,
            epochs=epochs,
            hidden_factor=hidden_factor,
            val_percentage=val_percentage
        )
        print(f"Results saved to: {save_path}")
    
    return results

def _save_results_to_csv(results, dataset, freeze_decoder, **kwargs):
    """
    Save results to CSV file following the original naming convention.
    """
    # Create directory based on dataset and freeze setting
    if dataset == 'single_solvent':
        results_dir = f"results/single_solv_decoder{freeze_decoder}"
    else:  # full_data
        results_dir = f"results/full_data_decoder{freeze_decoder}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate unique filename
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{timestamp}_{unique_id}.csv"
    save_path = os.path.join(results_dir, filename)
    
    # Create row for CSV
    row = {
        "timestamp": timestamp,
        "pretrained_model_path": kwargs.get('pretrained_model_path', ''),
        "spange_path": kwargs.get('spange_path', ''),
        "nn_size": kwargs.get('nn_size', 16),
        "dropout_fp": kwargs.get('dropout_fp', 0.1),
        "dropout_nn": kwargs.get('dropout_nn', 0.1),
        "epochs": kwargs.get('epochs', 10),
        "hidden_factor": kwargs.get('hidden_factor', 2),
        "val_percent": kwargs.get('val_percentage', 0.2),
        "freeze_decoder": freeze_decoder,
        "dataset": dataset,
        "avg_mse": results["avg_mse"],
        "mse_per_solvent": json.dumps(str(results["mse_per_solvent"]))
    }
    
    pd.DataFrame([row]).to_csv(save_path, index=False)
    return save_path

def run_sodade_quick_test(dataset="full_data", freeze_decoder=False):
    """
    Run a quick test with minimal epochs for development/testing.
    """
    return run_sodade_regression(
        dataset=dataset,
        freeze_decoder=freeze_decoder,
        epochs=2,  # Quick test with minimal epochs
        save_results=False
    )

def run_sodade_optimal_single_solvent(freeze_decoder=False):
    """
    Run SoDaDE regression on single solvent dataset with optimal parameters.
    """
    return run_sodade_regression(
        dataset="single_solvent",
        freeze_decoder=freeze_decoder,
        nn_size=128,
        dropout_fp=0.05,
        dropout_nn=0.05,
        epochs=50,
        hidden_factor=4,
        val_percentage=0.2,
        learning_rate_fp=1e-5,
        learning_rate_nn=1e-4
    )

def run_sodade_optimal_full_data(freeze_decoder=False):
    """
    Run SoDaDE regression on full dataset with optimal parameters.
    """
    return run_sodade_regression(
        dataset="full_data",
        freeze_decoder=freeze_decoder,
        nn_size=64,
        dropout_fp=0.05,
        dropout_nn=0.05,
        epochs=50,
        hidden_factor=4,
        val_percentage=0.2,
        learning_rate_fp=1e-5,
        learning_rate_nn=1e-4
    )

if __name__ == "__main__":
    #Run quick test automatically when called from command line
    print("Running quick test (2 epochs, full_data dataset, decoder not frozen)...")
    results = run_sodade_quick_test(dataset="full_data", freeze_decoder=False)
    print(f"\nQuick test completed! Average MSE: {results['avg_mse']:.4f}")
    
    # Uncomment any of the lines below for other experiments:
    
    #Full run with optimal parameters for single solvent
    #print("\nRunning full single solvent experiment...")
    
    #print("\nRunning with custom parameters...")
    #results = run_sodade_regression(
        #dataset="full_data",
        #freeze_decoder=False,
        #nn_size=64,
        #dropout_fp=0.05,
        #dropout_nn=0.05,
        #epochs=50,
        #hidden_factor=4,
        #val_percentage=0.2,
        #learning_rate_fp=1e-5,
        #learning_rate_nn=1e-4
    #) 
    # # Full run with optimal parameters for full dataset with frozen decoder
    # print("\nRunning full data experiment with frozen decoder...")
    # results = run_sodade_optimal_full_data(freeze_decoder=True)
    
    # # Custom parameters example
    # print("\nRunning with custom parameters...")
    # results = run_sodade_regression(
    #     dataset="full_data",
    #     freeze_decoder=False,
    #     epochs=15,
    #     nn_size=32,
    #     dropout_fp=0.15,
    #     learning_rate_nn=5e-4
    # )