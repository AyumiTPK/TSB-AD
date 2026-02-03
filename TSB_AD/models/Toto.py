"""
This function is adapted from [toto]
Original source: [https://github.com/DataDog/toto/blob/main/toto/notebooks/inference_tutorial.ipynb]
"""

from __future__ import annotations

import numpy as np
import torch

from TSB_AD.models.base import BaseDetector

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


class TotoDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "Datadog/Toto-Open-Base-1.0",
        context_length: int = 4096,
        prediction_length: int = 336,
        num_samples: int = 256,
        samples_per_batch: int = 256,
        use_kv_cache: bool = True,
        batch_size: int = 1,
        normalize_per_window: bool = True,
        time_interval_seconds: int = 1,
        device: str | None = None,
        compile_model: bool = False,
    ):
        self.model_name = "Toto"
        self.model_path = model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.use_kv_cache = use_kv_cache
        self.batch_size = batch_size
        self.normalize_per_window = normalize_per_window
        self.time_interval_seconds = time_interval_seconds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compile_model = compile_model

        self.model = None
        self.forecaster = None
        self.decision_scores_ = None
        self._fitted = False

    def _load_model(self):
        print("[TotoDetector] _load_model: start")
        if self.model is None:
            print(
                f"[TotoDetector] Loading model from {self.model_path} on device={self.device}"
            )
            self.model = Toto.from_pretrained(self.model_path)
            self.model.to(self.device)
            if self.compile_model:
                print("[TotoDetector] Compiling model")
                self.model.compile()
        if self.forecaster is None:
            print("[TotoDetector] Initializing forecaster")
            self.forecaster = TotoForecaster(self.model.model)
        print("[TotoDetector] _load_model: done")

    def fit(self, data, y=None):
        # Zero-shot model; nothing to fit
        print("[TotoDetector] fit: zero-shot, nothing to fit")
        self._fitted = True
        return self

    def zero_shot(self, data):
        print("[TotoDetector] zero_shot: delegating to decision_function")
        return self.decision_function(data, use_pretrained=True)

    def decision_function(self, X, use_pretrained: bool = True):
        print(
            f"[TotoDetector] decision_function: X.shape={getattr(X, 'shape', None)}, "
            f"use_pretrained={use_pretrained}"
        )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (T, F), got shape {X.shape}")

        self._load_model()
        print(
            f"[TotoDetector] context_length={self.context_length}, "
            f"prediction_length={self.prediction_length}, batch_size={self.batch_size}"
        )

        data_win, data_target = self.create_dataset(
            X,
            slidingWindow=self.context_length,
            predict_time_steps=self.prediction_length,
        )
        print(
            f"[TotoDetector] create_dataset: data_win.shape={data_win.shape}, "
            f"data_target.shape={data_target.shape}"
        )

        if data_win.shape[0] == 0:
            print("[TotoDetector] No windows created; returning zeros")
            self.decision_scores_ = np.zeros(len(X), dtype=float)
            return self.decision_scores_

        if self.normalize_per_window:
            # data_win: (N, F, L)
            print("[TotoDetector] normalize_per_window=True")
            X_mean = data_win.mean(axis=2, keepdims=True)
            X_std = data_win.std(axis=2, keepdims=True) + 1e-8
            X_scaled = (data_win - X_mean) / X_std
            y_scaled = (data_target - X_mean) / X_std
        else:
            print("[TotoDetector] normalize_per_window=False")
            X_scaled = data_win
            y_scaled = data_target

        all_preds = []
        for i in range(0, len(X_scaled), self.batch_size):
            batch = X_scaled[i : i + self.batch_size]  # (B, F, L)
            print(
                f"[TotoDetector] Batch {i}:{i + self.batch_size}, batch.shape={batch.shape}"
            )

            series = torch.tensor(batch, dtype=torch.float32, device=self.device)
            print(
                f"[TotoDetector] series.shape={tuple(series.shape)}, device={series.device}"
            )
            padding_mask = torch.ones_like(series, dtype=torch.bool)
            id_mask = torch.zeros_like(series)

            timestamp_seconds = torch.zeros_like(series, dtype=torch.long)
            time_interval_seconds = torch.full(
                (series.shape[0], series.shape[1]),
                self.time_interval_seconds,
                dtype=torch.long,
                device=self.device,
            )
            print(
                f"[TotoDetector] time_interval_seconds={self.time_interval_seconds}, "
                f"time_interval_seconds.shape={tuple(time_interval_seconds.shape)}"
            )

            inputs = MaskedTimeseries(
                series=series,
                padding_mask=padding_mask,
                id_mask=id_mask,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
            )

            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=self.prediction_length,
                num_samples=self.num_samples,
                samples_per_batch=self.samples_per_batch,
                use_kv_cache=self.use_kv_cache,
            )
            print(
                f"[TotoDetector] forecast.samples.shape={tuple(forecast.samples.shape)}"
            )

            samples = forecast.samples  # (B, F, P, S)
            preds = torch.median(samples, dim=-1).values  # (B, F, P)
            preds = preds.permute(0, 2, 1).detach().cpu().numpy()  # (B, P, F)
            print(f"[TotoDetector] preds.shape={preds.shape}")
            all_preds.append(preds)

        preds = np.concatenate(all_preds, axis=0)  # (N, P, F)
        print(f"[TotoDetector] concatenated preds.shape={preds.shape}")

        scores = np.mean((y_scaled - preds) ** 2, axis=(1, 2))
        print(f"[TotoDetector] scores.shape={scores.shape}")

        pad_length = self.context_length + self.prediction_length - 1
        padded_scores = np.zeros(len(X), dtype=float)
        padded_scores[:pad_length] = scores[0]
        padded_scores[pad_length:] = scores
        print(
            f"[TotoDetector] padded_scores.shape={padded_scores.shape}, "
            f"pad_length={pad_length}"
        )

        self.decision_scores_ = padded_scores
        print("[TotoDetector] decision_function: done")
        return padded_scores

    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        print(
            f"[TotoDetector] create_dataset: X.shape={X.shape}, "
            f"slidingWindow={slidingWindow}, predict_time_steps={predict_time_steps}"
        )
        Xs, ys = [], []
        for i in range(len(X) - slidingWindow - predict_time_steps + 1):
            tmp = X[i : i + slidingWindow + predict_time_steps]  # (L+P, F)
            x = tmp[:slidingWindow].T  # (F, L)
            y = tmp[slidingWindow:].T  # (F, P)
            Xs.append(x)
            ys.append(y)
        Xs_arr = np.array(Xs)
        ys_arr = np.array(ys)
        print(
            f"[TotoDetector] create_dataset: Xs.shape={Xs_arr.shape}, "
            f"ys.shape={ys_arr.shape}"
        )
        return Xs_arr, ys_arr