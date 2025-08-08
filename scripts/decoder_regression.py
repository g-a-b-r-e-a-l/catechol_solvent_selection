import os
import pandas as pd
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)
from catechol import metrics
from catechol.models.decoder import Decoder  # assuming your Decoder class is saved in decoder.py

def train_decoder_once(pretrained_model_path, spange_path,
                       learning_rate_FP=1e-5,
                       learning_rate_NN=1e-4,
                       dropout_FP=0.1,
                       dropout_NN=0.1,
                       epochs=10):
    """
    Train and evaluate a Decoder model on leave-one-solvent-out splits.
    """

    # --- Load dataset ---
    single_solvent = load_single_solvent_data()
    X = single_solvent[["Residence Time", "Temperature", "Reaction SMILES", "SOLVENT SMILES", "SOLVENT NAME", "SOLVENT Ratio"]]
    Y = single_solvent[["SM", "Product 2", "Product 3"]]
    print(f"Loaded dataset with {len(X)} samples.")
    split_generator = generate_leave_one_out_splits(X, Y)
    mse_scores = []
    solvent_names = []

    # --- Initialize model ---
    model = Decoder(
        pretrained_model_path=pretrained_model_path,
        spange_path=spange_path,
        learning_rate_FP=learning_rate_FP,
        learning_rate_NN=learning_rate_NN,
        dropout_FP=dropout_FP,
        dropout_NN=dropout_NN,
        epochs=epochs,
        time_limit=10800,
        batch_size=16
    )
    print("Initialized Decoder model with the following parameters:")
    # --- Loop through leave-one-solvent-out splits ---
    for split_idx, ((train_X, train_Y), (test_X, test_Y)) in enumerate(split_generator, 1):
        print(f"\nSplit {split_idx}: training on {len(train_X)} samples, testing on {len(test_X)} samples.")

        # Train
        model._train(train_X, train_Y)

        # Prepare test set and evaluate
        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model._predict(test_X)
        print(predictions.columns)
        print(predictions.shape, 'PREDS SHAPE')
        print(predictions['SM mean'], test_Y['SM']) #NEED TO BOUND THE PREDICTIONS PROPERLY BECUASE YOU CAN'T HAVE NEGATIVE PRODUCT
        mse = metrics.mse(predictions, test_Y)

        solvent = test_X["SOLVENT NAME"].unique()[0]
        mse_scores.append(mse)
        solvent_names.append(solvent)

        print(f"  {solvent}: MSE = {mse:.4f}")

    avg_mse = sum(mse_scores) / len(mse_scores)
    print("\n--- Results ---")
    for solvent, mse in zip(solvent_names, mse_scores):
        print(f"{solvent}: {mse:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")

    return {
        "learning_rate_FP": learning_rate_FP,
        "learning_rate_NN": learning_rate_NN,
        "dropout_FP": dropout_FP,
        "dropout_NN": dropout_NN,
        "avg_mse": avg_mse,
        "mse_per_solvent": dict(zip(solvent_names, mse_scores))
    }

if __name__ == "__main__":
    # Example usage
    pretrained_model_path = "scripts/val_loss0.1097_DM_128_TL_2_heads_32.pth"
    spange_path = "scripts\spange_melt.csv"
    
    results = train_decoder_once(
        pretrained_model_path=pretrained_model_path,
        spange_path=spange_path,
        learning_rate_FP=1e-5,
        learning_rate_NN=1e-4,
        dropout_FP=0.1,
        dropout_NN=0.1,
        epochs=10
    )
    
    print("Training complete. Results:", results)