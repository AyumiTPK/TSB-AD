"""
This function is adapted from [chronos-forecasting]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""


import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from .base import BaseDetector

class Chronos2Detector(BaseDetector):
    def __init__(
        self,
        model_name="amazon/chronos-2",
        prediction_length=1,
        device=None,
        quantile_levels=[0.1, 0.5, 0.9],
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=self.device
        )
        self.decision_scores_ = None
        self._fitted = False

    def _compute_scores(self, actual, pred):
        scores = np.zeros(len(actual))
        if len(pred) < len(actual):
            # prediction is shorter by prediction_length - 1
            pad_len = len(actual) - len(pred)
            scores[:pad_len] = (actual[:pad_len] - pred[0]) ** 2
            scores[pad_len:] = (actual[pad_len:] - pred) ** 2
        else:
            scores[self.prediction_length:] = (actual[self.prediction_length:] - pred[:-self.prediction_length]) ** 2
            scores[:self.prediction_length] = scores[self.prediction_length]
        return scores

    def _convert_dataframe(self, data):
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=[f"target_{i}" for i in range(data.shape[1])])
            # If univariate, rename to "target"
            if df.shape[1] == 1:
                df = df.rename(columns={"target_0": "target"})
            return df
        return data.copy()

    def _ensure_id(self, df, id_column="_id"):
        if id_column not in df.columns:
            df[id_column] = 0
        return df, id_column

    def _ensure_timestamp(self, df, timestamp_column=None):
        if timestamp_column not in df.columns:
            df["_timestamp"] = np.arange(len(df))
            timestamp_column = "_timestamp"
        return df, timestamp_column

    def zero_shot(
        self,
        data: pd.DataFrame,
        future_covariates: pd.DataFrame = None,
        id_column: str = "_id",
        timestamp_column: str = None,
    ):

        df = self._convert_dataframe(data)
        df, id_column = self._ensure_id(df, id_column)
        df, timestamp_column = self._ensure_timestamp(df, timestamp_column)

        target_cols = [c for c in df.columns if c not in {id_column, timestamp_column}]

        pred_df = self.pipeline.predict_df(
            df=df,
            future_df=future_covariates,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target_cols,
        )

        scores_list = []
        for target in target_cols:
            actual = df[target].values
            pred = pred_df.loc[
                pred_df["target_name"] == target, "predictions"
            ].values
            padded_scores = self._compute_scores(actual, pred)
            scores_list.append(padded_scores)

        self.decision_scores_ = np.mean(np.array(scores_list), axis=0)
        return self.decision_scores_

    def _prepare_inputs(self, data, past_covariates=None, future_covariates=None, id_column="_id"):
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
        else:
            df = data.copy()

        train_inputs = []
        for item_id, group in df.groupby(id_column):
            input_dict = {"target": group["target"].values}

            # Past covariates
            if past_covariates:
                input_dict["past_covariates"] = {col: group[col].values for col in past_covariates}
            else:
                input_dict["past_covariates"] = {}

            # Future covariates
            if future_covariates:
                input_dict["future_covariates"] = {col: None for col in future_covariates}
            else:
                input_dict["future_covariates"] = {}

            train_inputs.append(input_dict)
        return train_inputs

    def fit(
        self,
        data: pd.DataFrame,
        past_covariates=None,
        future_covariates=None,
        id_column="_id",
        timestamp_column="timestamp",
        num_steps=1000,
        batch_size=32,
        learning_rate=1e-5,
        fine_tune=True,
    ):
        if fine_tune:
            df = self._convert_dataframe(data)
            df, id_column = self._ensure_id(df, id_column)

            train_inputs = self._prepare_inputs(
                df, past_covariates=past_covariates, future_covariates=future_covariates, id_column=id_column
            )

            self.pipeline = self.pipeline.fit(
                inputs=train_inputs,
                prediction_length=self.prediction_length,
                num_steps=num_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        self._fitted = True

    def decision_function(self, data, future_covariates=None, id_column="_id", timestamp_column="timestamp"):
        if not self._fitted:
            raise RuntimeError("Model must be fine-tuned before calling decision_function")

        df = self._convert_dataframe(data)
        df, id_column = self._ensure_id(df, id_column)
        df, timestamp_column = self._ensure_timestamp(df, timestamp_column)

        target_cols = [c for c in df.columns if c not in {id_column, timestamp_column}]

        print("self.quantile_levels:", self.quantile_levels)
        pred_df = self.pipeline.predict_df(
            df=df,
            future_df=future_covariates,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target_cols,
        )

        scores_list = []
        for target in target_cols:
            actual = df[target].values
            pred = pred_df.loc[pred_df["target_name"] == target, "predictions"].values
            scores_list.append(self._compute_scores(actual, pred))

        return np.mean(np.array(scores_list), axis=0)


