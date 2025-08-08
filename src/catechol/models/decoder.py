import copy
import os
import random
import time
import ast
import math
import torch.optim as optim
import torch.nn.functional as F  # Missing - used in MaskedMultiHeadSelfAttentionBlock


import numpy as np
import pandas as pd
import pkg_resources
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader


from catechol.data.data_labels import get_data_labels_mean_var
from catechol.data.loader import generate_leave_one_out_splits, train_test_split

from .base_model import Model

class Decoder(Model):
    def __init__(
        self,
        pretrained_model_path: str,
        spange_path: str,
        learning_rate_FP: float = 1e-5,
        learning_rate_NN: float = 1e-4,
        dropout_FP: float = 0.1,
        dropout_NN: float = 0.1,
        epochs: int = 10,
        time_limit: float = 10800,
        batch_size: int = 16,
        **kwargs,
    ):

        self.spange_path = spange_path
        self.lr_FP = learning_rate_FP
        self.lr_NN = learning_rate_NN
        self.dropout_FP = dropout_FP
        self.dropout_NN = dropout_NN
        self.epochs = epochs
        self.time_limit = time_limit
        self.training_start_time = None
        self.batch_size = batch_size


        self.config = Config()
        self.spange_featuriser = SpangeDataset(spange_path, self.config.column_dict,  self.config.word_tokens,  self.config.token_types,  self.config.max_sequence_length)
        self.chemberta_dimension = self.spange_featuriser.chemberta_dim()


        self.FP_model = self.load_FP_model(
            model_path=pretrained_model_path,
            chemberta_dimension=self.chemberta_dimension
        )


        self.NN_model = NeuralNetworkModel(
            input_dim=128,
            output_dim=3, 
            hidden_dim=16,  # Hidden dimension for the neural network
            hidden_dim_factor=2,  # Factor to scale the hidden dimension
            dropout_rate=dropout_NN,
        )

        self.FP_optimizer = optim.AdamW(self.FP_model.parameters(), lr=self.lr_FP)
        self.NN_optimizer = optim.AdamW(self.NN_model.parameters(), lr=self.lr_NN)
        self.criterion = nn.MSELoss()
        


    def load_data(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        data = pd.read_csv(data_path)
        return data
    
    def load_FP_model(self, model_path, chemberta_dimension):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model
            chemberta_dimension: Dimension of ChemBERTa features
            dropout_rate: Dropout rate used during training
        
        Returns:
            Loaded model
        """
        model = MultiModalRegressionTransformer(
            chemberta_fp_dim=chemberta_dimension,
            column_vocab_size=self.config.word_vocab_size,
            transformer_hidden_dim=128,
            max_sequence_length=self.config.max_sequence_length,
            token_type_vocab_size=self.config.token_type_vocab_size,
            num_attention_heads=32,
            num_transformer_layers=2,
            dropout_rate=self.dropout_FP
        )
        
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return model
    
    def train_test_split_and_zscore_normalize_solvent_leakage(self, df, smiles_sample):
        """
        Splits the DataFrame into training and validation sets by holding out
        all data points corresponding to 3 random 'SMILES' strings for validation.
        Then, it performs Z-score normalization based ONLY on the training data
        and applies the normalization to both sets.
        
        Args:
            data_path (str): The path to the CSV file to load.
        
        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: Normalized training set.
                - pd.DataFrame: Normalized validation set.
                - dict: A dictionary of Z-score normalization parameters (mean and std)
                        used for each numerical column, calculated from the training set.
        """
        # Assuming load_and_clean_csv is a function you have defined elsewhere
        
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or not loaded correctly.")
        
        if 'SMILES' not in df.columns:
            raise ValueError("The DataFrame must contain a 'SMILES' column.")

        smiles_strings = df['SMILES'].unique()
        
        # Check if there are enough unique SMILES strings to sample
        if len(smiles_strings) < smiles_sample:
            raise ValueError(
                f"Not enough unique SMILES strings to sample 3 for validation. Found only {len(smiles_strings)}."
            )

        # 1. Randomly select 3 SMILES strings to use for the validation set
        validation_smiles = np.random.choice(smiles_strings, size=smiles_sample, replace=False)
        
        # 2. Split the data based on the selected SMILES strings
        val_df = df[df['SMILES'].isin(validation_smiles)].copy()
        train_df = df[~df['SMILES'].isin(validation_smiles)].copy()
        
        # Check if the split resulted in non-empty dataframes
        if train_df.empty or val_df.empty:
            raise ValueError("The split resulted in an empty training or validation set. Try running again.")

        # 3. Identify numerical columns based on the training data
        # Exclude 'SMILES' as it is a categorical identifier
        numerical_cols = train_df.select_dtypes(include=np.number).columns
        
        if numerical_cols.empty:
            print("No numerical columns found for normalization.")
            return train_df, val_df, {}
            
        normalization_params = {}
        
        # 4. Calculate mean and std deviation ONLY from the TRAINING set
        for col in numerical_cols:
            col_mean = train_df[col].mean()
            col_std = train_df[col].std(ddof=0)
            
            # Store the parameters
            normalization_params[col] = {'mean': col_mean, 'std': col_std}

            # 5. Normalize BOTH the training and validation sets using these parameters
            if col_std == 0:
                train_df[col] = 0.0
                val_df[col] = 0.0
            else:
                train_df.loc[:, col] = (train_df[col] - col_mean) / col_std
                val_df.loc[:, col] = (val_df[col] - col_mean) / col_std
                
        return train_df, val_df, normalization_params
    
    def normalize_or_unnormalize_df(self, df, normalization_params, command):
        """
        Normalizes or unnormalizes a pandas DataFrame using pre-calculated
        mean and standard deviation values.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            normalization_params (dict): A dictionary containing the normalization
                                        parameters for each column, e.g.,
                                        {'col_name': {'mean': ..., 'std': ...}}.
            command (str): The command to perform, either 'normalize' or 'unnormalize'.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # Make a copy of the DataFrame to avoid modifying the original
        processed_df = df.copy()

        # Iterate through the columns for which we have normalization parameters
        for col, params in normalization_params.items():
            if col in processed_df.columns:
                col_mean = params['mean']
                col_std = params['std']

                if command == 'normalize':
                    # Handle case where standard deviation is 0 to avoid division by zero
                    if col_std != 0:
                        processed_df.loc[:, col] = (processed_df[col] - col_mean) / col_std
                    else:
                        processed_df.loc[:, col] = 0.0
                elif command == 'unnormalize':
                    processed_df.loc[:, col] = (processed_df[col] * col_std) + col_mean
                else:
                    print(f"Warning: Command '{command}' not recognized. Column '{col}' was not processed.")

        return processed_df
    
    def collate_fn(self, batch):
        # Stack inputs from batch
        batch_df = pd.concat(batch)
        targets = []
        smiles_column = self.column_dict['SMILES_COLUMNS']
        ratio_column = self.column_dict['RATIO']
        res_temp_cols = self.column_dict['RES_TEMP_COLUMNS']
        targets_col = self.column_dict['YIELD_COLUMNS']

        smiles = batch_df[smiles_column].to_list()
        ratios = batch_df[ratio_column].to_list()

        res_t = batch_df[res_temp_cols[0]].to_list()
        temp = batch_df[res_temp_cols[1]].to_list()
        for i in targets_col:
            if i in batch_df.columns:
                targets.append(batch_df[i].to_list())

            else:
                # If the target column is missing, fill with NaNs
                targets.append([float('nan')] * len(batch_df))
                print(f"Warning: Column '{i}' not found in the DataFrame. Filling with NaNs.")

        res_time = torch.tensor(res_t, dtype=torch.float32)
        temp = torch.tensor(temp, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).T 
        
        return {
            'res_time' : res_time,
            'temp' : temp,
            'smiles' : smiles, # Contains original values, including NaNs where applicable.
            'ratios': ratios,
            'targets': targets, # Contains MASK_TOKEN for MLM-masked positions, original types otherwise.
        }        
    
    def generate_fp(self, model, batch_smiles, ratios, featuriser):
        look_up = {}
        count = 0
        total_smiles = []
        for index, smiles in enumerate(batch_smiles):
            indicies = []
            look_up[index] = {}
            line_smiles = smiles.split('.')

            look_up[index]['smiles'] = line_smiles
            look_up[index]['ratio'] = ast.literal_eval(ratios[index])
            for i in range(len(look_up[index]['ratio'])):
                total_smiles.append(line_smiles[i])
                indicies.append(count)
                count += 1
            look_up[index]['tensor_indices'] = indicies

        sequence_dict = featuriser.decoder_input_tensor(total_smiles)
        
        SMILES_fps = sequence_dict['SMILES_fps']
        word_tokens_ref = sequence_dict['word_tokens_ref']
        values_ref = sequence_dict['values_ref']
        token_type_ids = sequence_dict['token_type_ids']
        attention_mask = sequence_dict['attention_mask']
        # Run generative inference
        predictions = model(     
            token_type_vocab=self.config.token_type_vocab,
            SMILES_fps=SMILES_fps,
            word_tokens_ref=word_tokens_ref,
            values_ref=values_ref,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        batch_fps = []
        
        for batch_index in range(len(batch_smiles)):
            fp_tensors = []
            for fp_index in range(len(look_up[batch_index]['tensor_indices'])):
                smiles_location = look_up[batch_index]['tensor_indices'][fp_index]
                ratio = look_up[batch_index]['ratio'][fp_index]
                relative_print = predictions[smiles_location, -1, :] * float(ratio)
                # Fix 1: Use append instead of extend for tensor list
                fp_tensors.append(relative_print.unsqueeze(0))
                    
            # Fix 2: Stack tensors first, then sum along the correct dimension
            fps = torch.cat(fp_tensors, dim=0)  # Shape: [num_components, fp_dim]
            
            # Fix 3: Sum along dimension 0 to combine components
            fp = torch.sum(fps, dim=0)  # Shape: [fp_dim]
            
            # Fix 4: Use append instead of extend for batch list
            batch_fps.append(fp.unsqueeze(0))  # Add batch dimension: [1, fp_dim]
        
        # Final concatenation along batch dimension
        batch_fps = torch.cat(batch_fps, dim=0)  # Shape: [batch_size, fp_dim]
        
        return batch_fps

    def _train(self, train_X, train_Y):
        data_df = pd.concat([train_X, train_Y], axis=1)
        train_df, val_df, self.normalization_params = self.train_test_split_and_zscore_normalize_solvent_leakage(data_df, 
                                                                                                            smiles_sample=3)
        
        train_dataset = MolecularPropertyDataset(train_df, self.config.column_dict)
        val_dataset = MolecularPropertyDataset(val_df, self.config.column_dict)

        dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        dataloader_val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        self.training_start_time = time.time()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time))}")
        print(f"Time limit: {self.time_limit/3600:.1f} hours ({self.time_limit} seconds)")

        
        best_val_loss = float('inf')
        self.best_FP_model_state = None
        self.best_NN_model_state = None
        for epoch in range(self.epochs):
            self.FP_model.train()
            self.NN_model.train()

            
            epoch_loss = 0

            for batch_idx, batch_train in enumerate(tqdm(dataloader_train, desc=f"Training Epoch {epoch + 1}/{self.epochs}")):
                # CHECK TIME LIMIT EVERY 10 BATCHES (adjust frequency as needed)
                if batch_idx % 10 == 0:
                    elapsed_time = time.time() - self.training_start_time
                    if elapsed_time >= self.time_limit:
                        self.early_stop_reason = f"Time limit reached during training batch {batch_idx}"
                        print(f"\n⏰ {self.early_stop_reason}")
                        break
                # Extract inputs and targets from the batch
                res_time = batch_train['res_time']
                temp = batch_train['temp']
                smiles = batch_train['smiles']
                ratios = batch_train['ratios']
                targets = batch_train['targets']

                self.FP_optimizer.zero_grad()
                self.NN_optimizer.zero_grad()


                solvent_fp = self.generate_fp(self.FP_model, smiles, ratios, self.spange_featuriser)
                # Forward pass through the neural network model
                yield_predictions = self.NN_model(solvent_fp, res_time, temp)

                train_loss = self.criterion(yield_predictions, targets)
                train_loss.backward()
                self.FP_optimizer.step()
                self.NN_optimizer.step()

                epoch_loss += train_loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader_train)

            #validation phase
            self.FP_model.eval()
            self.NN_model.eval()
            epoch_val_loss = 0
            unnorm_epoch_val_loss = 0
            target_cols = self.config.column_dict['YIELD_COLUMNS']
            output_means = torch.tensor([self.normalization_params[col]['mean'] for col in target_cols])
            output_stds = torch.tensor([self.normalization_params[col]['std'] for col in target_cols])
            with torch.no_grad():
                for batch_val in tqdm(dataloader_val, desc=f"Validation Epoch {epoch + 1}/{self.epochs}"):
                    res_time = batch_val['res_time']
                    temp = batch_val['temp']
                    smiles = batch_val['smiles']
                    ratios = batch_val['ratios']
                    targets = batch_val['targets']

                    solvent_fp = self.generate_fp(self.FP_model, smiles, ratios, self.spange_featuriser)
                    yield_predictions = self.NN_model(solvent_fp, res_time, temp)

                    val_loss = self.criterion(yield_predictions, targets)
                    epoch_val_loss += val_loss.item()

                    unnorm_predictions = (yield_predictions * output_stds) + output_means
                
                    # Unnormalize targets
                    unnorm_targets = (targets * output_stds) + output_means
                    
                    # Calculate loss on the unnormalized values
                    unnorm_val_loss = self.criterion(unnorm_predictions, unnorm_targets)
                    unnorm_epoch_val_loss += unnorm_val_loss.item()


            avg_epoch_val_loss = epoch_val_loss / len(dataloader_val)
            unnorm_avg_epoch_val_loss = unnorm_epoch_val_loss / len(dataloader_val)

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                self.best_FP_model_state = copy.deepcopy(self.FP_model.state_dict())
                self.best_NN_model_state = copy.deepcopy(self.NN_model.state_dict())
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")


    def _predict(self, test_X):
        #load best performing model of fp and NN
        self.FP_model.load_state_dict(self.best_FP_model_state)
        self.NN_model.load_state_dict(self.best_NN_model_state)
        text_X_norm = self.normalize_or_unnormalize_df(test_X, self.normalization_params, 'normalize')
        test_dataset = MolecularPropertyDataset(text_X_norm, self.config.column_dict)
        dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

        self.FP_model.eval()
        self.NN_model.eval()
        predictions = []
        with torch.no_grad():
            for batch_val in tqdm(dataloader_test, desc="Testing"):
                res_time = batch_val['res_time']
                temp = batch_val['temp']
                smiles = batch_val['smiles']
                ratios = batch_val['ratios']
                targets = batch_val['targets']

                solvent_fp = self.generate_fp(self.FP_model, smiles, ratios, self.spange_featuriser)
                yield_predictions = self.NN_model(solvent_fp, res_time, temp)
                predictions.append(yield_predictions)
        predictions = torch.cat(predictions, dim=0)
        pred_df = pd.DataFrame(predictions.cpu().numpy(), columns=self.config.target_labels)
        # Unnormalize predictions
        predictions = self.normalize_or_unnormalize_df(pred_df, self.normalization_params, 'unnormalize')
                
                
        return predictions
    

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Shape (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (embedded sequence).
                               Shape: (batch_size, sequence_length, d_model)
        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class MaskedMultiHeadSelfAttentionBlock(nn.Module):
    """
    A single block for a decoder-only Transformer model,
    implementing masked multi-head self-attention and a feed-forward network.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=False
        )
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        ff_output = self.linear2(self.dropout_ffn(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ff_output)
        tgt = self.norm2(tgt)
        return tgt
    
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim_factor: int):
        super(FeedForwardNeuralNetwork, self).__init__()

        ff_dimension = hidden_dim_factor * output_dim

        # First linear layer: 10 inputs, 20 outputs
        self.fc1 = nn.Linear(input_dim, ff_dimension)
        # Activation function
        self.relu = nn.ReLU()
        # Second linear layer: 4 * chemberta dimension inputs, 5 outputs
        self.fc2 = nn.Linear(ff_dimension, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiModalInputEmbeddings(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, dropout_rate: float):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.smiles_proj = FeedForwardNeuralNetwork(chemberta_fp_dim, transformer_hidden_dim, 4)
        self.property_embedding = nn.Embedding(column_vocab_size, transformer_hidden_dim)
        self.value_proj = nn.Linear(1, transformer_hidden_dim)
        self.token_type_embeddings = nn.Embedding(token_type_vocab_size, transformer_hidden_dim)
        self.position_encodings = PositionalEncoding(transformer_hidden_dim, dropout_rate, max_sequence_length)
        self.LayerNorm = nn.LayerNorm(transformer_hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                token_type_vocab: dict,
                SMILES_fps: torch.Tensor,
                word_tokens_ref: torch.Tensor,
                values_ref: torch.Tensor,
                token_type_ids: torch.Tensor,
                ):
        batch_size, max_batch_seq_len = token_type_ids.shape
        input_embeddings = torch.zeros(batch_size, max_batch_seq_len, self.transformer_hidden_dim,
                                       dtype=torch.float, device=token_type_ids.device)


        word_mask = (token_type_ids == token_type_vocab['WORD_TOKEN'])
        smiles_mask = (token_type_ids == token_type_vocab['SMILES_TOKEN'])
        value_mask = (token_type_ids == token_type_vocab['VALUE_TOKEN'])

        if word_mask.any():
            input_embeddings[word_mask] = self.property_embedding(word_tokens_ref[word_mask])
        if smiles_mask.any():
            input_embeddings[smiles_mask] = self.smiles_proj(SMILES_fps[smiles_mask])


        if value_mask.any():
            input_embeddings[value_mask] = self.value_proj(values_ref[value_mask].unsqueeze(-1))

        token_type_embedding_values = self.token_type_embeddings(token_type_ids)
        embeddings = input_embeddings + token_type_embedding_values
        embeddings = self.position_encodings(embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class MultiModalRegressionTransformer(nn.Module):
    def __init__(self, chemberta_fp_dim: int, column_vocab_size: int,
                 transformer_hidden_dim: int, max_sequence_length: int,
                 token_type_vocab_size: int, num_attention_heads: int,
                 num_transformer_layers: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = transformer_hidden_dim
        self.embeddings_module = MultiModalInputEmbeddings(
            chemberta_fp_dim=chemberta_fp_dim,
            column_vocab_size=column_vocab_size,
            transformer_hidden_dim=transformer_hidden_dim,
            max_sequence_length=max_sequence_length,
            token_type_vocab_size=token_type_vocab_size,
            dropout_rate=dropout_rate
        )
        self.transformer_decoder_layers = nn.ModuleList([
            MaskedMultiHeadSelfAttentionBlock(
                d_model=transformer_hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=transformer_hidden_dim * 4,
                dropout=dropout_rate
            )
            for _ in range(num_transformer_layers)
        ])
        self.regression_head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 4 * transformer_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(4 * transformer_hidden_dim),
            nn.Linear(4 * transformer_hidden_dim, 1)
        )
        self._init_weights()


    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                token_type_vocab: dict,
                SMILES_fps : torch.Tensor,
                word_tokens_ref : torch.Tensor,
                values_ref : torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                ) -> torch.Tensor:
        batch_size, sequence_length = token_type_ids.shape
        embeddings = self.embeddings_module(
            token_type_vocab=token_type_vocab,
            SMILES_fps=SMILES_fps,
            word_tokens_ref=word_tokens_ref,
            values_ref=values_ref,
            token_type_ids=token_type_ids
        )
        causal_mask = self.generate_square_subsequent_mask(sequence_length).to(embeddings.device)
        tgt_key_padding_mask = ~attention_mask
        transformer_output = embeddings
        for layer in self.transformer_decoder_layers:
            transformer_output = layer(
                tgt=transformer_output,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        return transformer_output

    def generative_inference(self,
                           token_type_vocab: dict,
                           SMILES_fps: torch.Tensor,
                           word_tokens_ref: torch.Tensor,
                           values_ref: torch.Tensor,
                           token_type_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           positions_to_predict: list):
        """
        Generates missing values in a sequence one by one.

        Args:
            ... (same as forward pass) ...
            positions_to_predict (list): A list of indices in the sequence where
                                         values need to be predicted.
        """
        self.eval() # Set the model to evaluation mode
        with torch.no_grad():
            # Clone the inputs to avoid modifying the originals
            values_ref_filled = values_ref.clone()

            for pos in sorted(positions_to_predict):
                # Run a forward pass with the current state of the inputs
                transformer_output = self.forward(
                    token_type_vocab=token_type_vocab,
                    SMILES_fps=SMILES_fps,
                    word_tokens_ref=word_tokens_ref,
                    values_ref=values_ref_filled,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                )

                # Get the hidden state for the token we want to predict a value for
                # transformer_output shape: (batch_size, sequence_length, hidden_dim)
                hidden_state_to_predict = transformer_output[:, pos, :]

                # Pass this hidden state through the regression head
                predicted_value = self.regression_head(hidden_state_to_predict).squeeze(-1)

                # Update the values_ref tensor with the predicted value
                # This ensures the model uses this new information for the next prediction
                values_ref_filled[:, pos] = predicted_value

        return values_ref_filled
    
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim:int, hidden_dim_factor: int, dropout_rate: int):
        super(NeuralNetworkModel, self).__init__()

        self.fp_projection = FeedForwardNeuralNetwork(input_dim, hidden_dim, hidden_dim_factor)
        self.res_time_projection = nn.Linear(1, hidden_dim)
        self.temp_projection = nn.Linear(1, hidden_dim)
        print('hidden dimension:', hidden_dim, 'input dim', input_dim, 'output dim ', output_dim)

        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

        self.yield_pred_head = FeedForwardNeuralNetwork(hidden_dim, output_dim, hidden_dim_factor)

    def forward(self, spange_fp, res_time, temp):
        fp_projection = self.fp_projection(spange_fp)
        res_time_projection = self.res_time_projection(res_time.unsqueeze(1))
        temp_projection = self.temp_projection(temp.unsqueeze(1))

        fp_projection = self.LayerNorm(fp_projection)
        res_time_projection = self.LayerNorm(res_time_projection)
        temp_projection = self.LayerNorm(temp_projection)

        fp_projection = self.dropout(fp_projection)
        res_time_projection = self.dropout(res_time_projection)
        temp_projection = self.dropout(temp_projection)

        combination = fp_projection + res_time_projection + temp_projection
        yields = self.yield_pred_head(combination)

        return yields    
    
class MolecularPropertyDataset(Dataset):
    def __init__(self, df, column_dict
                 ):
        self.df = df
        



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        

        return self.df.iloc[idx]
    
    
    

class SpangeDataset():
    def __init__(self, spange_path, column_dict, word_tokens, token_types, max_sequence_length=28):
        self.utils = Utils()
        self.spange_df =  pd.read_csv(spange_path)
        self.column_dict = column_dict
        self.max_sequence_length = max_sequence_length
        self.word_tokens = word_tokens
        self.token_types = token_types
        self.tokenizer, self.model_chemberta, self.chemberta_dimension = self.utils.load_chemberta_model_and_tokenizer()

    def chemberta_dim(self):
        return self.chemberta_dimension

    def smiles_lookup(self, batch_smiles):
        """
        Looks up SMILES strings from batch_smiles in a larger DataFrame
        and returns a new DataFrame with rows ordered according to batch_smiles,
        preserving the original column order of spange_df.

        Args:
            spange_path (str): Path to the CSV file containing the master data.
            batch_smiles (list): A list of SMILES strings to look up and order by.
            column_dict (dict): A dictionary containing column names, e.g., {'SMILES_COLUMNS': 'SMILES'}.

        Returns:
            pd.DataFrame: A DataFrame containing rows from spange_df,
                        reordered according to batch_smiles, and with the
                        same column order as the original spange_df.
                        Rows for SMILES not found in spange_df will be excluded.
        """
        smiles_col = self.column_dict['SMILES_COLUMNS']

        # Store the original column order of spange_df
        original_columns_order = self.spange_df.columns.tolist()

        # Create a DataFrame from batch_smiles to use for merging and maintaining order
        batch_df_order = pd.DataFrame({smiles_col: batch_smiles})

        # Merge spange_df with batch_df_order.
        # An 'inner' merge is used to keep only SMILES present in both,
        # and the order will be preserved from batch_df_order.
        merged_df = pd.merge(batch_df_order, self.spange_df, on=smiles_col, how='inner')

        # Reindex the merged DataFrame to match the original column order of spange_df
        # This will ensure all columns from spange_df are present and in the correct order.
        # If any column from original_columns_order is not in merged_df (e.g., if it was a column
        # created only by the merge itself, which isn't the case here with `how='inner'`),
        # it would be filled with NaNs.
        spange_desc_df = merged_df[original_columns_order]

        return spange_desc_df

    def shuffle_column_pairs(self, df):
        """
        For each row, shuffles its property-value pairs once while
        keeping the row order the same. Each row is returned once.

        Args:
            df (pd.DataFrame): The input DataFrame with 'solvent', 'SMILES',
                            and interleaved 'Property_i' and 'Value_i' columns.

        Returns:
            pd.DataFrame: A new DataFrame with each original row containing
                        its property-value pairs shuffled once.
        """
        generated_rows = []

        # Define the fixed identifier columns
        id_cols = ['solvent', 'SMILES']

        # Dynamically find all property-value column indices
        # We use a set for efficiency and then sort for consistent order
        pair_indices = sorted(list(set([int(col.split('_')[1]) for col in df.columns if col.startswith('Property_')])))

        if not pair_indices:
            print("Error: No 'Property_i' columns found. Returning an empty DataFrame.")
            return pd.DataFrame(columns=df.columns)

        # Prepare column order for the final DataFrame
        column_order = id_cols
        for i in pair_indices:
            column_order.append(f'Property_{i}')
            column_order.append(f'Value_{i}')

        # Iterate over each original row in the DataFrame
        for _, original_row in df.iterrows():
            # Collect all property-value pairs for the current row
            property_value_pairs = []
            for i in pair_indices:
                prop_col = f'Property_{i}'
                val_col = f'Value_{i}'
                prop = original_row.get(prop_col, np.nan)
                val = original_row.get(val_col, np.nan)
                # Only add if both property and value are not NaN, or handle as per desired logic
                # For this context, we'll keep them even if NaN, as they are part of a pair
                property_value_pairs.append((prop, val))

            # Shuffle the pairs for the current row
            random.shuffle(property_value_pairs)

            # Create a new row dictionary, starting with identifier columns
            new_row_data = {col: original_row[col] for col in id_cols}

            # Populate the new row with the shuffled property-value pairs
            for i, (prop, val) in enumerate(property_value_pairs):
                # Assign to dynamic Property_i and Value_i based on the shuffled order
                # This ensures that Property_0, Value_0, etc., contain the *shuffled* pairs
                new_row_data[f'Property_{i}'] = prop
                new_row_data[f'Value_{i}'] = val
            
            generated_rows.append(new_row_data)

        # Create the final DataFrame from the list of generated row dictionaries
        if not generated_rows:
            return pd.DataFrame(columns=column_order) # Return with correct columns even if empty

        final_df = pd.DataFrame(generated_rows)
        # Reindex to ensure consistent column order with original df structure
        final_df = final_df.reindex(columns=column_order)

        return final_df


    def decoder_input(self, spange_desc_df):

        word_columns = self.column_dict['WORD_COLUMNS']
        value_columns = self.column_dict['VALUE_COLUMNS']
        smiles_column = self.column_dict['SMILES_COLUMNS']
        
        token_type_vocab, word_vocab = self.utils.create_diff_token_vocabs(self.word_tokens, self.token_types)

        values_ref, missing_val_mask = self.utils.create_values_tensor(spange_desc_df, self.max_sequence_length, value_columns)
        word_tokens_ref = self.utils.word_token_indicies(spange_desc_df, self.max_sequence_length, word_columns, word_vocab)
        token_type_ids = self.utils.create_token_type_tensor(spange_desc_df, self.max_sequence_length, self.column_dict, token_type_vocab)

        SMILES_fps = self.utils.create_fingerprints(spange_desc_df, self.max_sequence_length, smiles_column, self.tokenizer, self.model_chemberta)

        attention_mask = torch.ones(token_type_ids.shape, dtype=torch.bool)

        # 2. Apply the 'missing_val_mask' to the attention_mask.
        # These positions should be ignored by attention as they contain no information.
        # The 'token_type_ids' for these positions remain unchanged (e.g., VALUE_TOKEN).
        attention_mask[missing_val_mask] = False
        # 3. Initialize masked_lm_labels to ignore index (-100.0)
        # This is a common convention for PyTorch's CrossEntropyLoss to ignore loss at these positions
        
        token_type_ids[missing_val_mask] = token_type_vocab['MASK_TOKEN'] 
        return {
            'values_ref': values_ref,
            'missing_val_mask': missing_val_mask,
            'word_tokens_ref': word_tokens_ref,
            'token_type_ids': token_type_ids,
            'SMILES_fps': SMILES_fps,
            'attention_mask': attention_mask
        }

    def decoder_input_tensor(self, batch_smiles):
        """
        Returns the tensors for the dataset, which can be used in a DataLoader.
        This method is useful for creating batches of data for training or inference.
        """
        spange_desc_df = self.smiles_lookup(batch_smiles)
        spange_shuffle_desc = self.shuffle_column_pairs(spange_desc_df)

        return self.decoder_input(spange_shuffle_desc)
    
class Utils():

    def load_chemberta_model_and_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            model_chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            model_chemberta.eval() # Set to evaluation mode for fingerprint generation
            chemberta_fp_dimension = model_chemberta.config.hidden_size
            print("ChemBERTa loaded successfully with hidden size:", chemberta_fp_dimension)
        except Exception as e:
            print(f"Error loading ChemBERta: {e}")
            exit()

        return tokenizer, model_chemberta, chemberta_fp_dimension

    def create_values_tensor(self, df, max_seq_length, value_columns):
        values_tensor = torch.full((df.shape[0], max_seq_length), -100, dtype=torch.float)

        col_id = 1 # account for start token column
        for i in df.columns:
            if i in value_columns:
                values_tensor[:, col_id] = torch.tensor(df[i])
            col_id += 1

        missing_val_mask = torch.isnan(values_tensor)

        return values_tensor, missing_val_mask

    def create_diff_token_vocabs(self, WORD_TOKENS, TOKEN_TYPES):
        word_vocab = {col: i for i, col in enumerate(WORD_TOKENS)}
        token_type_vocab = {token_type: i for i, token_type in enumerate(TOKEN_TYPES)}

        return token_type_vocab, word_vocab


    def word_token_indicies(self, df, max_seq_length, word_columns, word_vocab):
        word_index_tensor = torch.zeros((df.shape[0], max_seq_length), dtype=torch.long)
        col_id = 1 # account for start token column
        for i in df.columns:
            if i in word_columns:
                word_index_tensor[:, col_id] = torch.tensor(df[i].map(word_vocab))
            col_id += 1
        return word_index_tensor

    def create_fingerprints(self, df, max_seq_length, smiles_column, tokenizer, model_chemberta):
        smiles_list = df[smiles_column].tolist()
        chemberta_fp_dim = model_chemberta.config.hidden_size
        SMILES_fps = torch.zeros((df.shape[0], max_seq_length, chemberta_fp_dim), dtype=torch.float)

        # Tokenize SMILES strings
        smiles_tokenized_inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            # Move inputs to device if you have a GPU for model_chemberta
            # smiles_tokenized_inputs = {k: v.to(device) for k, v in smiles_tokenized_inputs.items()}
            outputs = model_chemberta(**smiles_tokenized_inputs, output_hidden_states=True)
            smiles_chemberta_fps = outputs.hidden_states[-1].mean(dim=1) # Shape: (batch_size, CHEMBE`RTA_FP_DIM)

        SMILES_fps[:, 2, :] = smiles_chemberta_fps

        return SMILES_fps

    def create_token_type_tensor(self, df, max_seq_length, column_dict, token_type_vocab):
        tensor_dict = {}

        for i in list(token_type_vocab.keys()):
            tensor_dict[i] = torch.full((df.shape[0], ), token_type_vocab[i], dtype=torch.int)

        token_type_tensor = torch.zeros(df.shape[0], max_seq_length, dtype=torch.int)

        token_type_tensor[:, 0] = tensor_dict['CLS_TOKEN']
        token_type_tensor[:, -1] = tensor_dict['SEP_TOKEN']
        tensor_index = 0
        for i in range(df.shape[1]):
            tensor_index = i+1 #starts at 1 becuase we have added a CLS column beforehand
            if df.columns[i] in column_dict['WORD_COLUMNS']:
                token_type_tensor[:, tensor_index] = tensor_dict['WORD_TOKEN']
            elif df.columns[i] in column_dict['VALUE_COLUMNS']:
                token_type_tensor[:, tensor_index] = tensor_dict['VALUE_TOKEN']
            elif df.columns[i] == column_dict['SMILES_COLUMNS']:
                token_type_tensor[:, tensor_index] = tensor_dict['SMILES_TOKEN']
            else:
                pass

        return token_type_tensor
    
class Config():

    def __init__(self):
        self.column_dict = {
        'WORD_COLUMNS': ['solvent', 'Property_0', 'Property_1', 'Property_2',  'Property_3',  'Property_4', 'Property_5',
                            'Property_6', 'Property_7', 'Property_8', 'Property_9', 'Property_10',  'Property_11'],
        'VALUE_COLUMNS': ['Value_0', 'Value_1', 'Value_2', 'Value_3', 'Value_4', 'Value_5',
                            'Value_6','Value_7', 'Value_8', 'Value_9', 'Value_10', 'Value_11'],
        'SMILES_COLUMNS': 'SMILES', 
        'RATIO': 'SOLVENT Ratio',
        'RES_TEMP_COLUMNS': ['Residence_Time', 'Temperature'], 
        'YIELD_COLUMNS': ['SM', 'Product_2', 'Product_3']}

        self.token_types = ['WORD_TOKEN', 'SMILES_TOKEN', 'VALUE_TOKEN', 'MASK_TOKEN', 'CLS_TOKEN', 'SEP_TOKEN']
        self.word_tokens = ['alkane', 'aromatic', 'halohydrocarbon', 'ether', 'ketone', 'ester', 'nitrile', 'amine', 'amide', 'misc_N_compound', 'carboxylic_acid', 'monohydric_alcohol' , 'polyhydric_alcohol', 'other','ET30', 'alpha', 'beta', 'pi_star', 'SA', 'SB', 'SP', 'SdP', 'N_mol_cm3', 'n', 'fn', 'delta']
        self.token_type_vocab = {token_type: i for i, token_type in enumerate(self.token_types)}
        self.max_sequence_length = 28  # Maximum sequence length for the model
        self.word_vocab_size = len(self.word_tokens)
        self.token_type_vocab_size = len(self.token_types)
        self.target_labels = self.column_dict['YIELD_COLUMNS']




    



