"""
This function is adapted from [Moirai]
Original source: [https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/moirai_forecast.ipynb]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset

from TSB_AD.models.base import BaseDetector

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class MoiraiDetector(BaseDetector):
    def __init__(
        self,
        model_size: str = "small",
        model_path: str | None = None,
        context_length: int = 1024,
        prediction_length: int = 96,
        patch_size: int | str = "auto",
        num_samples: int = 100,
        batch_size: int = 32,
        normalize_per_window: bool = True,
        freq: str = "1H",
    ):
        self.model_name = "Moirai"
        self.model_size = model_size
        self.model_path = model_path or f"Salesforce/moirai-1.1-R-{model_size}"
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.normalize_per_window = normalize_per_window
        self.freq = freq

        self.model = None
        self.predictor = None
        self._target_dim = None
        self.decision_scores_ = None
        self._fitted = False

    def _load_model(self, target_dim: int):
        print(f"[MoiraiDetector] _load_model: target_dim={target_dim}")
        if self.model is None or self._target_dim != target_dim:
            print(
                f"[MoiraiDetector] Loading model from {self.model_path} "
                f"(model_size={self.model_size})"
            )
            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(self.model_path),
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=target_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            print(
                f"[MoiraiDetector] Creating predictor with batch_size={self.batch_size}"
            )
            self.predictor = self.model.create_predictor(batch_size=self.batch_size)
            self._target_dim = target_dim
        print("[MoiraiDetector] _load_model: done")

    def fit(self, data, y=None):
        print("[MoiraiDetector] fit: zero-shot, nothing to fit")
        self._fitted = True
        return self

    def zero_shot(self, data):
        print("[MoiraiDetector] zero_shot: delegating to decision_function")
        return self.decision_function(data, use_pretrained=True)

    def decision_function(self, X, use_pretrained: bool = True):
        print(
            f"[MoiraiDetector] decision_function: X.shape={getattr(X, 'shape', None)}, "
            f"use_pretrained={use_pretrained}"
        )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (T, F), got shape {X.shape}")

        print(
            f"[MoiraiDetector] context_length={self.context_length}, "
            f"prediction_length={self.prediction_length}, batch_size={self.batch_size}"
        )
        data_win, data_target = self.create_dataset(
            X,
            slidingWindow=self.context_length,
            predict_time_steps=self.prediction_length,
        )
        print(
            f"[MoiraiDetector] create_dataset: data_win.shape={data_win.shape}, "
            f"data_target.shape={data_target.shape}"
        )

        if data_win.shape[0] == 0:
            print("[MoiraiDetector] No windows created; returning zeros")
            self.decision_scores_ = np.zeros(len(X), dtype=float)
            return self.decision_scores_

        # data_win: (N, F, L), data_target: (N, F, P)
        if self.normalize_per_window:
            print("[MoiraiDetector] normalize_per_window=True")
            X_mean = data_win.mean(axis=2, keepdims=True)
            X_std = data_win.std(axis=2, keepdims=True) + 1e-8
            X_scaled = (data_win - X_mean) / X_std
            y_scaled = (data_target - X_mean) / X_std
        else:
            print("[MoiraiDetector] normalize_per_window=False")
            X_scaled = data_win
            y_scaled = data_target

        n_windows, n_features, _ = X_scaled.shape
        print(
            f"[MoiraiDetector] n_windows={n_windows}, n_features={n_features}"
        )
        self._load_model(target_dim=n_features)

        all_preds = []
        start_ts = pd.Timestamp("2000-01-01")

        for i in range(0, n_windows, self.batch_size):
            batch = X_scaled[i : i + self.batch_size]  # (B, F, L)
            print(
                f"[MoiraiDetector] Batch {i}:{i + self.batch_size}, batch.shape={batch.shape}"
            )

            dataset = ListDataset(
                [
                    {
                        "start": start_ts,
                        "target": batch[j],
                    }
                    for j in range(batch.shape[0])
                ],
                freq=self.freq,
            )
            print(
                f"[MoiraiDetector] ListDataset created with freq={self.freq}"
            )

            forecasts = list(self.predictor.predict(dataset))
            print(f"[MoiraiDetector] forecasts count={len(forecasts)}")

            # Each forecast.samples: (num_samples, P, F)
            batch_preds = []
            for fc in forecasts:
                samples = fc.samples  # (S, P, F)
                print(
                    f"[MoiraiDetector] forecast.samples.shape={samples.shape}"
                )
                median = np.median(samples, axis=0)  # (P, F)
                batch_preds.append(median.T)  # (F, P)

            all_preds.append(np.stack(batch_preds, axis=0))  # (B, F, P)
            print(
                f"[MoiraiDetector] batch_preds.shape={all_preds[-1].shape}"
            )

        preds = np.concatenate(all_preds, axis=0)  # (N, F, P)
        print(f"[MoiraiDetector] concatenated preds.shape={preds.shape}")

        scores = np.mean((y_scaled - preds) ** 2, axis=(1, 2))
        print(f"[MoiraiDetector] scores.shape={scores.shape}")

        pad_length = self.context_length + self.prediction_length - 1
        padded_scores = np.zeros(len(X), dtype=float)
        padded_scores[:pad_length] = scores[0]
        padded_scores[pad_length:] = scores
        print(
            f"[MoiraiDetector] padded_scores.shape={padded_scores.shape}, "
            f"pad_length={pad_length}"
        )

        self.decision_scores_ = padded_scores
        print("[MoiraiDetector] decision_function: done")
        return padded_scores

    def create_dataset(self, X, slidingWindow, predict_time_steps=1):
        print(
            f"[MoiraiDetector] create_dataset: X.shape={X.shape}, "
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
            f"[MoiraiDetector] create_dataset: Xs.shape={Xs_arr.shape}, "
            f"ys.shape={ys_arr.shape}"
        )
        return Xs_arr, ys_arr