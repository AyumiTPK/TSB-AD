"""
This function is adapted from [chronos-forecasting]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""

import numpy as np
import pandas as pd
import torch
from chronos.chronos2.pipeline import BaseChronosPipeline, Chronos2Pipeline
from .base import BaseDetector

class Chronos2(BaseDetector):
    def __init__(
        self,
        model_name="amazon/chronos-2",
        context_length=512, 
        prediction_length=1,
        device=None,
        quantile_levels=[0.1, 0.5, 0.9],
        validation_size=0.2,
        num_steps=1000,
        batch_size=32,
        learning_rate=1e-5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.context_length = context_length 
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels

        self._finetune_params = {
            "validation_size": validation_size,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }

        self.pretrained_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=self.device
        )
        self.pipeline = self.pretrained_pipeline
        self.decision_scores_ = None
        self._fitted = False

    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - slidingWindow - predict_time_steps + 1):
            tmp = X[i : i + slidingWindow + predict_time_steps]  # Shape: (slidingWindow + predict_time_steps, num_variates)
            x = tmp[:slidingWindow].T  # Shape: (num_variates, slidingWindow)
            y = tmp[slidingWindow:].T  # Shape: (num_variates, predict_time_steps)
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)

    def zero_shot(self, data):
        print(f"[DEBUG] zero_shot called with data shape: {data.shape}")
        self.decision_scores_ = self.decision_function(data, use_pretrained=True)

    def fit(self, data, y=None):
        print(f"[DEBUG] fit called with data shape: {data.shape}")
        
        validation_size = self._finetune_params.get("validation_size", 0.2)
        num_steps = self._finetune_params.get("num_steps", 1000)
        batch_size = self._finetune_params.get("batch_size", 32)
        learning_rate = self._finetune_params.get("learning_rate", 1e-5)
        
        if len(data) < self.context_length + self.prediction_length:
            print(f"Data too short ({len(data)} samples). Skipping fine-tuning.")
            return

        split_idx = int((1 - validation_size) * len(data))
        data_train = data[:split_idx]
        data_valid = data[split_idx:]
        print(f"[DEBUG] Train data: {data_train.shape} ({len(data_train)/len(data)*100:.1f}%)")
        print(f"[DEBUG] Valid data: {data_valid.shape} ({len(data_valid)/len(data)*100:.1f}%)")
        
        data_win, data_target = self.create_dataset(
            data_train, 
            slidingWindow=self.context_length, 
            predict_time_steps=self.prediction_length
        )
        if len(data_win) < 10:
            print(f"Too few training windows ({len(data_win)}). Skipping fine-tuning.")
            return
        print(f"[DEBUG] Created {data_win.shape[0]} training windows from training data")

        train_inputs = [{"target": data_win[i]} for i in range(len(data_win))]
        
        self.pipeline = self.pretrained_pipeline.fit(
            inputs=train_inputs,
            prediction_length=self.prediction_length,
            num_steps=num_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        self._fitted = True

    def decision_function(self, X, use_pretrained=False):
        if use_pretrained:
            pipeline = self.pretrained_pipeline
            print(f"[DEBUG] Using pretrained model")
        else:
            if not self._fitted:
                print(f"Model not fine-tuned. Using pretrained model.")
                pipeline = self.pretrained_pipeline
            else:
                pipeline = self.pipeline
                print(f"[DEBUG] Using fine-tuned model")
        
        return self._compute_scores(X, pipeline)
        
    def _compute_scores(self, data, pipeline):
        print(f"[DEBUG] data shape: {data.shape}")
        print(f"[DEBUG] context_length: {self.context_length}, prediction_length: {self.prediction_length}")
        data_win, data_target = self.create_dataset(
            data, 
            slidingWindow=self.context_length, 
            predict_time_steps=self.prediction_length
        )
        print(f"[DEBUG] Created {data_win.shape[0]} windows for scoring")
        print(f"[DEBUG] data_win shape: {data_win.shape}, data_target shape: {data_target.shape}")
        
        quantiles, mean = pipeline.predict_quantiles(
            inputs=data_win,
            prediction_length=self.prediction_length,
            quantile_levels=[0.5],
        )
        
        predictions = np.array(mean)
        print(f"[DEBUG] predictions shape: {predictions.shape}")
        
        scores = np.mean((data_target - predictions) ** 2, axis=(1, 2))
        print(f"[DEBUG] scores shape: {scores.shape}")
        print(f"[DEBUG] scores min/max/mean: {scores.min():.4f} / {scores.max():.4f} / {scores.mean():.4f}")
        
        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.context_length + self.prediction_length - 1] = scores[0]
        padded_decision_scores[self.context_length + self.prediction_length - 1:] = scores
        
        print(f"[DEBUG] padded_decision_scores shape: {padded_decision_scores.shape}")
        print(f"[DEBUG] padded_decision_scores min/max: {padded_decision_scores.min():.4f} / {padded_decision_scores.max():.4f}\n")
        
        self.decision_scores_ = padded_decision_scores
        return self.decision_scores_
