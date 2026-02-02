"""
This function is adapted from [TTMs]
Original source: [https://github.com/ibm-granite/granite-tsfm]
"""

import os
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

from TSB_AD.models.base import BaseDetector
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions
from tsfm_public.toolkit.lr_finder import optimal_lr_finder 
import math

class TTM(BaseDetector):
    def __init__(self,
                 model_path="ibm-granite/granite-timeseries-ttm-r2",
                 context_length=512,
                 prediction_length=96,
                 batch_size=4,
                 num_epochs=50,
                 learning_rate=0.001,
                 freeze_backbone=False,
                 loss="mse",
                 quantile=0.5,
                 validation_size=0.1):
        
        self.model_name = 'TTM'
        self.model_path = model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        self.loss = loss
        self.quantile = quantile

        self._finetune_params = {
            "validation_size": validation_size,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_backbone": freeze_backbone,
        }

        self.pretrained_model = None
        self.model = None
        self.tsp = None
        self.column_specifiers = {}
        self.split_config = {}
        self.decision_scores_ = None
        self._fitted = False

    def zero_shot(self, data):
        return self.decision_function(data, use_pretrained=True)

    def fit(self, data, y=None):
        print(f"[FT] Input data shape: {data.shape}")

        validation_size = self._finetune_params.get("validation_size", 0.1)
        num_epochs = self._finetune_params.get("num_epochs", 50)
        batch_size = self._finetune_params.get("batch_size", 4)
        learning_rate = self._finetune_params.get("learning_rate", 0.001)
        freeze_backbone = self._finetune_params.get("freeze_backbone", False)

        num_features = data.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]
        df = pd.DataFrame(data, columns=feature_names)
        print(f"[FT] DataFrame shape: {df.shape}")

        if not self.column_specifiers:
            self.column_specifiers = {
                "timestamp_column": None,
                "id_columns": [],
                "target_columns": feature_names,
                "control_columns": [],
            }

        if not self.split_config:
            num_rows = len(df)
            train_end = int(0.8 * num_rows)                    
            valid_end = int((0.8 + validation_size) * num_rows) 
            
            self.split_config = {
                "train": [0, train_end],              
                "valid": [train_end, valid_end],      
                "test": [valid_end, num_rows],        
            }
        print(f"[FT] Split config: train={self.split_config['train']}, valid={self.split_config['valid']}, test={self.split_config['test']}")

        self.tsp = TimeSeriesPreprocessor(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",
            column_specifiers=self.column_specifiers
        )

        print("[FT] Loading model")
        self.model = get_model(
            self.model_path,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            freq_prefix_tuning=False,
            freq=None,
            prefer_l1_loss=False,
            prefer_longer_context=True,
            loss=self.loss,
            quantile=self.quantile,
        )

        dset_train, dset_val, dset_test = get_datasets(
            self.tsp,
            df,
            self.split_config,
            fewshot_fraction=1.0, 
            fewshot_location="first",
            use_frequency_token=self.model.config.resolution_prefix_tuning
        )
        print(f"[FT] Dataset sizes: train={len(dset_train)}, val={len(dset_val)}, test={len(dset_test)}")

        if freeze_backbone:
            print("[FT] Number of params before freezing:", count_parameters(self.model))
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("[FT] Number of params after freezing:", count_parameters(self.model))

        if learning_rate is None:
            learning_rate, self.model = optimal_lr_finder(
            self.model,
            dset_train,
            batch_size=batch_size,
            )
            print("[FT] OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)
        else:
            print(f"[FT] Using provided learning rate: {learning_rate}")

        print("[FT] Training")
        temp_dir = tempfile.mkdtemp()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            report_to="none",
            seed=7,
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=8,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=1e-5,
        )
        tracking_callback = TrackingCallback()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=num_epochs,
            steps_per_epoch=math.ceil(len(dset_train) / batch_size),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),
        )
        trainer.train()
        self._fitted = True

    def decision_function(self, X, use_pretrained=False):
        print(f"[Decision] Called with data shape: {X.shape}, use_pretrained={use_pretrained}")

        if not use_pretrained and self._fitted and self.model is not None:
            model = self.model
            print("[Decision] Using fine-tuned model")
        else:
            print("[Decision] Using pretrained model")
            if self.pretrained_model is None:
                self.pretrained_model = get_model(
                    self.model_path,
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    freq_prefix_tuning=False,
                    prefer_longer_context=True,
                )
            model = self.pretrained_model

        data_win, data_target = self.create_dataset(
            X,
            slidingWindow=self.context_length,
            predict_time_steps=self.prediction_length
        )
        print(f"[Decision] Created {data_win.shape[0]} windows") 
        
        num_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]

        if use_pretrained or not self._fitted:
            print("[Decision] Using per-window normalization (zero-shot)")
            X_mean = data_win.mean(axis=1, keepdims=True)
            X_std = data_win.std(axis=1, keepdims=True) + 1e-8
            X_scaled = (data_win - X_mean) / X_std
            y_scaled = (data_target - X_mean) / X_std
        else:
            print("[Decision] Using fitted scaler (fine-tuned)")
            if not self.tsp.target_scaler_dict:
                raise RuntimeError("Target scaler is not trained; call fit() first.")
            scaler = self.tsp.target_scaler_dict.get("0")
            if scaler is None:
                scaler = next(iter(self.tsp.target_scaler_dict.values()))      
            print(f"[Decision] Scaler type: {type(scaler)}") 
             
            X_scaled = scaler.transform(
                pd.DataFrame(
                    data_win.reshape(-1, num_features),
                    columns=feature_names
                )
            ).reshape(data_win.shape)
            
            y_scaled = scaler.transform(
                pd.DataFrame(
                    data_target.reshape(-1, num_features),
                    columns=feature_names
                )
            ).reshape(data_target.shape)
        
        print(f"[Decision] X_scaled shape: {X_scaled.shape}, min/max: {X_scaled.min():.4f}/{X_scaled.max():.4f}")
        print(f"[Decision] y_scaled shape: {y_scaled.shape}, min/max: {y_scaled.min():.4f}/{y_scaled.max():.4f}")

        #if self.tsp is None:
        #    if not self.column_specifiers:
        #        self.column_specifiers = {
        #            "timestamp_column": None,
        #            "id_columns": [],
        #            "target_columns": feature_names,
        #            "control_columns": [],
        #        }
        #    self.tsp = TimeSeriesPreprocessor(
        #        context_length=self.context_length,
        #        prediction_length=self.prediction_length,
        #        scaling=True,
        #        encode_categorical=False,
        #        scaler_type="standard",
        #        column_specifiers=self.column_specifiers,
        #    )
            #df_full = pd.DataFrame(X, columns=feature_names)
            #self.tsp.train(df_full)
#
#        if not self.tsp.target_scaler_dict:
#            raise RuntimeError("Target scaler is not trained; call self.tsp.train(df_full) first.")
#        scaler = self.tsp.target_scaler_dict.get("0")
#        if scaler is None:
#            scaler = next(iter(self.tsp.target_scaler_dict.values()))
#        print(f"[Decision] Scaler type: {type(scaler)}")
#        X_scaled = scaler.transform(
#            data_win.reshape(-1, num_features)
#        ).reshape(data_win.shape)
#        print(f"[Decision] X_scaled shape: {X_scaled.shape}, min/max: {X_scaled.min():.4f}/{X_scaled.max():.4f}")
#
#        y_scaled = scaler.transform(
#            data_target.reshape(-1, num_features)
#        ).reshape(data_target.shape)
#        print(f"[Decision] y_scaled shape: {y_scaled.shape}, min/max: {y_scaled.min():.4f}/{y_scaled.max():.4f}")
        model.eval()
        device = next(model.parameters()).device
        print(f"[Decision] Model device: {device}")

        batch_size = 32
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch = torch.tensor(
                    X_scaled[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=device
                )
                outputs = model(past_values=batch)
                all_preds.append(outputs.prediction_outputs.detach().cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        print(f"[Decision] preds shape: {preds.shape}, min/max: {preds.min():.4f}/{preds.max():.4f}")
        #with torch.no_grad():
        #    outputs = model(
        #        past_values=torch.tensor(X_scaled, dtype=torch.float32, device=device)
        #    )
        #preds = outputs.prediction_outputs.detach().cpu().numpy()
        #print(f"[Decision] preds shape: {preds.shape}, min/max: {preds.min():.4f}/{preds.max():.4f}")

        scores = np.mean((y_scaled - preds) ** 2, axis=(1, 2))
        print(f"[Decision] scores shape: {scores.shape}, min/max/mean: {scores.min():.4f}/{scores.max():.4f}/{scores.mean():.4f}")

        pad_length = self.context_length + self.prediction_length - 1
        padded_scores = np.zeros(len(X))
        padded_scores[:pad_length] = scores[0]
        padded_scores[pad_length:] = scores
        print(f"[Decision] padded_scores shape: {padded_scores.shape}, expected: {len(X)}")
        print(f"[Decision] pad_length: {pad_length}, num_scores: {len(scores)}")

        self.decision_scores_ = padded_scores
        return padded_scores

    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingWindow - predict_time_steps + 1):
            tmp = X[i : i + slidingWindow + predict_time_steps]
            x = tmp[:slidingWindow]
            y = tmp[slidingWindow:]
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)