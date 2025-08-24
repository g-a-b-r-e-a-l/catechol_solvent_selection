import os
import pandas as pd
import argparse
import json
import uuid
from datetime import datetime
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
    load_solvent_ramp_data, generate_leave_one_ramp_out_splits
)
from catechol import metrics
from decoder import Decoder as Single_Decoder
from decoder_full_yields import Decoder as Full_Decoder


def train_decoder_once(dataset, pretrained_model_path, spange_path,
                       freeze_fp = False,
                       learning_rate_FP=1e-5,
                       learning_rate_NN=1e-4,
                       dropout_FP=0.1,
                       dropout_NN=0.1,
                       val_percentage=0.2,
                       NN_size=16,
                       hidden_factor=2,
                       epochs=10):
    """
    Train and evaluate a Decoder model on leave-one-solvent-out splits.
    """

    # --- Load dataset ---
    
    if dataset == 'single_solvent':
        single_solvent = load_single_solvent_data()
        X = single_solvent[[
            "Residence Time", "Temperature", "Reaction SMILES",
            "SOLVENT SMILES", "SOLVENT NAME", "SOLVENT Ratio"
        ]]
        Y = single_solvent[["SM", "Product 2", "Product 3"]]
        print(f"Loaded {dataset} dataset with {len(X)} samples.")
        split_generator = generate_leave_one_out_splits(X, Y)

    elif dataset == 'full_yields':
        data, targets = load_solvent_ramp_data()
        print(data.columns, targets.columns)
        X = data[[
            "Residence Time", "Temperature", 'SolventB%', "SOLVENT A NAME", "SOLVENT B NAME",
            "SOLVENT A SMILES", "SOLVENT B SMILES", "SOLVENT A Ratio", "SOLVENT B Ratio", "RAMP NUM"
        ]]
        Y = targets[["SM", "Product 2", "Product 3"]]
        print(f"Loaded {dataset} dataset with {len(X)} samples.")
        split_generator = generate_leave_one_ramp_out_splits(X, Y)

    else:
        if dataset != 'full_yields' or 'single_solvent':
            print("ERROR: DATASET SHOULD BE EITHER 'single_solvent' or 'full_yields'" )
            raise ValueError

    
    
    mse_scores = []
    test_items = []

    # --- Initialize model ---
    if dataset == 'single_solvent':
        model = Single_Decoder(
            pretrained_model_path=pretrained_model_path,
            spange_path=spange_path,
            freeze_fp=freeze_fp,
            learning_rate_FP=learning_rate_FP,
            learning_rate_NN=learning_rate_NN,
            dropout_FP=dropout_FP,
            dropout_NN=dropout_NN,
            val_percentage=val_percentage, 
            NN_size=NN_size,
            hidden_factor=hidden_factor,
            epochs=epochs,
            time_limit=10800,
            batch_size=16
        )

    elif dataset == 'full_yields':
        model = Full_Decoder(
            pretrained_model_path=pretrained_model_path,
            spange_path=spange_path,
            freeze_fp=freeze_fp,
            learning_rate_FP=learning_rate_FP,
            learning_rate_NN=learning_rate_NN,
            dropout_FP=dropout_FP,
            dropout_NN=dropout_NN,
            val_percentage=val_percentage, 
            NN_size=NN_size,
            hidden_factor=hidden_factor,
            epochs=epochs,
            time_limit=10800,
            batch_size=16
        )

    else:
        if dataset != 'full_yields' or 'single_solvent':
            print("ERROR: DATASET SHOULD BE EITHER 'single_solvent' or 'full_yields'" )
            raise ValueError
    
    print(f"Initialized {dataset} Decoder model with the following parameters:")
    print(f"  NN_size: {NN_size}, Hidden_Factor: {hidden_factor}, Epochs: {epochs}")
    print(f"  Dropout_FP: {dropout_FP}, Dropout_NN: {dropout_NN}")
    
    # --- Loop through leave-one-solvent-out splits ---
    for split_idx, ((train_X, train_Y), (test_X, test_Y)) in enumerate(split_generator, 1):
        print(f"\nSplit {split_idx}: training on {len(train_X)} samples, testing on {len(test_X)} samples.")

        # Train
        model._train(train_X, train_Y)

        # Prepare test set and evaluate
        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model._predict(test_X)
        mse = metrics.mse(predictions, test_Y)


        if dataset == 'single_solvent':

            test_item = test_X["SOLVENT NAME"].unique()[0]

        elif dataset == 'full_yields':
            test_item = float(test_X["RAMP NUM"].unique()[0])

        mse_scores.append(float(mse))
        test_items.append(test_item)

        print(f" Test Item is {test_item}: MSE = {mse:.4f}")

    avg_mse = float(sum(mse_scores) / len(mse_scores))
    print("\n--- Results ---")
    for test_item, mse in zip(test_items, mse_scores):
        print(f"{test_item}: {mse:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")

    return {
        "NN_size": NN_size,
        "hidden_factor": hidden_factor,
        "epochs": epochs,
        "dropout_FP": dropout_FP,
        "dropout_NN": dropout_NN,
        "avg_mse": avg_mse,
        "mse_per_solvent": dict(zip(test_items, mse_scores))
    }

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train and evaluate a Decoder model with hyperparameter tuning.")
    
    # File Paths
    parser.add_argument('--dataset', type=str,
                        default="single_solvent",
                        help='single_sovlent or full_yields')
    parser.add_argument('--pretrained_model_path', type=str,
                        default="val_loss0.1074_DPR_0.1_MP_0.3_DM_64_TL_5_heads_16.pth",
                        help='Path to the pretrained model file.')
    parser.add_argument('--spange_path', type=str,
                        default="spange_melt.csv",
                        help='Path to the spange data file.')
    
    # Grid Search Hyperparameters
    parser.add_argument('--nn_size', type=int, default=16, help='Size of the final neural network layer.')
    parser.add_argument('--dropout_fp', type=float, default=0.1, help='Dropout value for the fingerprint model.')
    parser.add_argument('--dropout_nn', type=float, default=0.1, help='Dropout value for the final neural network.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--hidden_factor', type=int, default=2, help='Hidden factor for the model architecture.')
    parser.add_argument('--val_percent', type=float, default=0.2, help='train test split percentage')

    
    # Learning Rates
    parser.add_argument('--freeze_fp_learning', action='store_true', help='Freeze the learning rate of the fingerprint model')
    parser.add_argument('--lr_fp', type=float, default=1e-5, help='Learning rate for the fingerprint model.')
    parser.add_argument('--lr_nn', type=float, default=1e-4, help='Learning rate for the neural network.')

    args = parser.parse_args()
    dataset = args.dataset
    freeze_fp=args.freeze_fp_learning

    print(dataset)
    # --- Call Training Function ---
    results = train_decoder_once(
        dataset=args.dataset,
        pretrained_model_path=args.pretrained_model_path,
        spange_path=args.spange_path,
        freeze_fp=args.freeze_fp_learning,
        learning_rate_FP=args.lr_fp,
        learning_rate_NN=args.lr_nn,
        dropout_FP=args.dropout_fp,
        dropout_NN=args.dropout_nn,
        val_percentage=args.val_percent, 
        NN_size=args.nn_size,
        hidden_factor=args.hidden_factor,
        epochs=args.epochs
    )
    
    print("\n--- Final Run Summary ---")
    print(results)

    # --- Save per-run results ---
    if args.dataset == 'single_solvent':
        os.makedirs(f"results/single_solv_decoder{freeze_fp}", exist_ok=True)
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}_{unique_id}.csv"
        save_path = os.path.join(f"results/single_solv_decoder{freeze_fp}", filename)

    elif args.dataset == 'full_yields':
        os.makedirs(f"results/full_data_decoder{freeze_fp}", exist_ok=True)
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}_{unique_id}.csv"
        save_path = os.path.join(f"results/full_data_decoder{freeze_fp}", filename)

    else:
        print('SOMETHING WAS SKIPPED')

    row = {
        "timestamp": timestamp,
        "pretrained_model_path": args.pretrained_model_path,
        "spange_path": args.spange_path,
        "nn_size": args.nn_size,
        "dropout_fp": args.dropout_fp,
        "dropout_nn": args.dropout_nn,
        "epochs": args.epochs,
        "hidden_factor": args.hidden_factor,
        "val_percent":args.val_percent,
        "avg_mse": results["avg_mse"],
        "mse_per_solvent": json.dumps(str(results["mse_per_solvent"]))
    }




    pd.DataFrame([row]).to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
