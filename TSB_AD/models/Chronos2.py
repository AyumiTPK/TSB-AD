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

    #def _compute_scores(self, actual, pred):
        #scores = (actual - pred) ** 2
        #padded_scores = np.zeros(len(actual))

        #pad_start = self.prediction_length - 1
        #start_pad_len = min(pad_start, len(actual))
        #padded_scores[:start_pad_len] = scores[0]  # pad start
        #padded_scores[start_pad_len:start_pad_len + len(scores)] = scores
        #if start_pad_len + len(scores) < len(actual):
        #    padded_scores[start_pad_len + len(scores):] = scores[-1]  # pad end

        #return padded_scores

    def _compute_scores(self, actual, pred):
        scores = (actual - pred) ** 2
        padded_scores = np.zeros(len(actual))

        pad_len = self.prediction_length - 1 # number of initial timesteps that cannot be scored
        padded_scores[:pad_len] = scores[0]  # pad the beginning with the first available score
        padded_scores[pad_len:] = scores # fill the rest with actual scores

        return padded_scores

    def zero_shot(
        self,
        data: pd.DataFrame,
        #past_covariates: pd.DataFrame = None,
        future_covariates: pd.DataFrame = None,
        id_column: str = None,
        timestamp_column: str = None,
        predict_batches_jointly: bool = False,
    ):

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame or np.ndarray, got {type(data)}")

        df = data.copy()

        if timestamp_column is None:
            df["_timestamp"] = np.arange(len(df))
            timestamp_column_used = "_timestamp"
        else:
            timestamp_column_used = timestamp_column
        if id_column is None:
            df["_id"] = 0
            id_column_used = "_id"
        else:
            id_column_used = id_column

        exclude_cols = {timestamp_column_used, id_column_used}
        target_cols = [c for c in df.columns if c not in exclude_cols]

        pred_df = self.pipeline.predict_df(
            df=df,
            future_df=future_covariates,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            id_column=id_column_used,
            timestamp_column=timestamp_column_used,
            target=target_cols,
            predict_batches_jointly=predict_batches_jointly,
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

    def _prepare_inputs(self, data, past_covariates=None, future_covariates=None, id_column="item_id"):
        train_inputs = []

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame or np.ndarray, got {type(data)}")

        df = data.copy()
        print(type(df))

        group_iter = [(None, data)] if id_column is None else data.groupby(id_column)
        for item_id, group in group_iter:
            input_dict = {"target": group["target"].values}

            # Past covariates
            if past_covariates:
                input_dict["past_covariates"] = {col: group[col].values for col in past_covariates}
            else:
                input_dict["past_covariates"] = None

            # Future covariates
            if future_covariates:
                input_dict["future_covariates"] = {col: None for col in future_covariates}
            else:
                input_dict["future_covariates"] = None

            train_inputs.append(input_dict)
        return train_inputs

    def fit(
        self,
        data: pd.DataFrame,
        past_covariates=None,
        future_covariates=None,
        id_column="item_id",
        timestamp_column="timestamp",
        num_steps=1000,
        batch_size=32,
        learning_rate=1e-5,
        fine_tune=True,
    ):
        if fine_tune:
            train_inputs = self._prepare_inputs(
                data, past_covariates=past_covariates, future_covariates=future_covariates, id_column=id_column
            )
            self.pipeline = self.pipeline.fit(
                inputs=train_inputs,
                prediction_length=self.prediction_length,
                num_steps=num_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        # Predict
        target_cols = [col for col in data.columns if col not in [id_column, timestamp_column]]
        pred_df = self.pipeline.predict_df(
            df=data,
            future_df=future_covariates,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target_cols,
        )

        # Compute anomaly scores
        scores_list = []
        for target in target_cols:
            actual = data[target].values
            pred = pred_df.loc[
                pred_df["target_name"] == target, "predictions"
            ].values
            padded_scores = self._compute_scores(actual, pred)
            scores_list.append(padded_scores)

        self.decision_scores_ = np.mean(np.array(scores_list), axis=0)

    def decision_function(self, X=None):
        """Return anomaly scores computed during fit or zero-shot."""
        return self.decision_scores_
