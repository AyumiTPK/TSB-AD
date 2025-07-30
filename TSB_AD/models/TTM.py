"""
This function is adapted from [TTMs] by [Ekambaram et al.]
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
                 context_length=512, #512,
                 prediction_length=96,#96,
                 batch_size=32,#64
                 num_epochs=50,
                 learning_rate=None,
                 fewshot_percent=5,
                 freeze_backbone=False,
                 loss="mse",
                 quantile=0.5):
        self.model_name = 'TTM'
        self.model_path = model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.fewshot_percent = fewshot_percent
        self.freeze_backbone = freeze_backbone
        self.loss = loss
        self.quantile = quantile
        self.model = None
        self.tsp = None
        self.column_specifiers = {}
        self.split_config = {}

    def zero_shot(self, data):

        print("[Zero] Reconstructing DataFrame")
        num_features = data.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]
        df = pd.DataFrame(data, columns=feature_names)

        if not self.column_specifiers:
            self.column_specifiers = {
                "timestamp_column": None,
                "id_columns": [],
                "target_columns": feature_names,
                "control_columns": [],
                #"conditional_columns": [],
            }

        if not self.split_config:
            num_rows = len(df)
            self.split_config = {
                "train": [0, int(0.8 * num_rows)],
                "valid": [int(0.8 * num_rows), int(0.9 * num_rows)],
                "test": [int(0.9 * num_rows), num_rows],
            }

        print("[Zero] Initializing preprocessor")
        self.tsp = TimeSeriesPreprocessor(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=True,
            #encode_categorical=False,
            scaler_type="standard",
            column_specifiers=self.column_specifiers
        )

        print("[Zero] Loading model")
        self.model = get_model(
            self.model_path,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            freq_prefix_tuning=False,
            freq=None,
            prefer_l1_loss=False,
            prefer_longer_context=True,
            #loss=self.loss,
            #quantile=self.quantile,
            #decoder_mode="mix_channel",
        )

        print("[Zero] Creating datasets")
        dset_train, dset_val, dset_test = get_datasets(
            self.tsp,
            df,
            self.split_config,
            use_frequency_token=self.model.config.resolution_prefix_tuning
        )

        print("[Zero] Training")
        temp_dir = tempfile.mkdtemp()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=self.batch_size,
            report_to="none",
            seed=7,
        )

        zeroshot_trainer = Trainer(
            model=self.model,
            args=training_args,
        )

        print("+" * 20, f"Test MSE zero-shot", "+" * 20)
        zeroshot_output = zeroshot_trainer.evaluate(dset_test)
        print(zeroshot_output)
        print("+" * 60)

        print("[Zero] Predicting")
        predictions = zeroshot_trainer.predict(dset_test)
        preds = predictions.predictions[0]

        print("[Zero] Extracting targets")
        targets = torch.stack([sample["future_values"] for sample in dset_test])

        preds = preds.numpy() if isinstance(preds, torch.Tensor) else preds
        targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets

        scores = (targets.squeeze() - preds.squeeze()) ** 2

        print("[Zero] Calculating mean squared error")
        per_time_feature_score = np.mean(scores, axis=1)

        if per_time_feature_score.ndim == 1:
            per_time_feature_score = per_time_feature_score[np.newaxis, :]

        pad_start = self.context_length + self.prediction_length - 1
        padded_score = np.zeros((len(data), num_features))
        padded_score[:pad_start, :] = per_time_feature_score[0]
        padded_score[pad_start:pad_start + len(per_time_feature_score), :] = per_time_feature_score

        print("[Zero] Padding complete")
        final_score = padded_score.mean(axis=1)
        self.decision_scores_ = final_score

    def fit(self, data):
        train_loss_history = []
        val_loss_history = []
        class LossTrackerCallback(TrackingCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    train_loss_history.append(logs['loss'])

            def on_evaluate(self, args, state, control, metrics, **kwargs):
                if 'eval_loss' in metrics:
                    val_loss_history.append(metrics['eval_loss'])

        print("[FT] Reconstructing DataFrame")
        num_features = data.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]
        df = pd.DataFrame(data, columns=feature_names)

        if not self.column_specifiers:
            self.column_specifiers = {
                "timestamp_column": None,
                "id_columns": [],
                "target_columns": feature_names,
                "control_columns": [],
                #"conditional_columns": [],
            }

        if not self.split_config:
            num_rows = len(df)
            self.split_config = {
                "train": [0, int(0.8 * num_rows)],
                "valid": [int(0.8 * num_rows), int(0.9 * num_rows)],
                "test": [int(0.9 * num_rows), num_rows],
            }

        print("[FT] Initializing preprocessor")
        self.tsp = TimeSeriesPreprocessor(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=True,
            #encode_categorical=False,
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
            #decoder_mode="mix_channel",
        )

        print("[FT] Creating datasets")
        dset_train, dset_val, dset_test = get_datasets(
            self.tsp,
            df,
            self.split_config,
            fewshot_fraction=self.fewshot_percent / 100,
            fewshot_location="first",
            use_frequency_token=self.model.config.resolution_prefix_tuning
        )
        #self.test_size_ = len(dset_test)

        if self.freeze_backbone:
            print("[FT] Freezing backbone parameters")
            print("Number of params before freezing:", count_parameters(self.model))

            for param in self.model.backbone.parameters():
                param.requires_grad = False

            print("Number of params after freezing:", count_parameters(self.model))

        if self.learning_rate is None:
            self.learning_rate, self.model = optimal_lr_finder(
            self.model,
            dset_train,
            batch_size=self.batch_size,
            )
            print("[FT] OPTIMAL SUGGESTED LEARNING RATE =", self.learning_rate)
        else:
            print(f"[FT] Using provided learning rate: {self.learning_rate}")

        print("[FT] Training")
        temp_dir = tempfile.mkdtemp()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=8,
            report_to="none",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=7,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=1e-5,
        )
        tracking_callback = TrackingCallback()

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=math.ceil(len(dset_train) / self.batch_size),
        )

        #trainer = Trainer(
        #    model=self.model,
        #    args=training_args,
        #    train_dataset=dset_train,
        #    eval_dataset=dset_val,
        #    callbacks=[early_stopping_callback, tracking_callback],
        #    optimizers=(optimizer, scheduler),
        #)

        loss_tracker = LossTrackerCallback()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            callbacks=[early_stopping_callback, tracking_callback, loss_tracker],
            optimizers=(optimizer, scheduler),
        )

        trainer.train()

        import matplotlib.pyplot as plt
        plt.plot(train_loss_history, label="Training Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss vs Epoch")
        plt.legend()
        plt.show()

        print("+" * 20, f"Test loss after fine-tuning", "+" * 20)
        trainer.model.loss = "mse"
        fewshot_output = trainer.evaluate(dset_test)
        print(fewshot_output)
        print("+" * 60)

        print("[FT] Predicting")
        predictions = trainer.predict(dset_test)
        preds = predictions.predictions[0]

        print("[FT] Extracting targets")
        targets = torch.stack([sample["future_values"] for sample in dset_test])

        preds = preds.numpy() if isinstance(preds, torch.Tensor) else preds
        targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets

        scores = (targets.squeeze() - preds.squeeze()) ** 2

        print("[FT] Calculating mean squared error")
        per_time_feature_score = np.mean(scores, axis=1)

        if per_time_feature_score.ndim == 1:
            per_time_feature_score = per_time_feature_score[np.newaxis, :]

        pad_start = self.context_length + self.prediction_length - 1
        padded_score = np.zeros((len(data), num_features))
        padded_score[:pad_start, :] = per_time_feature_score[0]
        padded_score[pad_start:pad_start + len(per_time_feature_score), :] = per_time_feature_score

        print("[FT] Padding complete")
        final_score = padded_score.mean(axis=1)
        self.decision_scores_ = final_score

    def decision_function(self, X):
        if not hasattr(self, 'decision_scores_') or self.decision_scores_ is None:
            raise RuntimeError("Scores not available. ")
        return self.decision_scores_







