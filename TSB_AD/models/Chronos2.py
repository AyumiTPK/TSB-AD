"""
This function is adapted from [chronos-forecasting]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""


import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from .base import BaseDetector

class Chronos2(BaseDetector):
    def __init__(
        self,
        model_name="amazon/chronos-2",
        context_length=512, 
        prediction_length=1,
        device=None,
        quantile_levels=[0.1, 0.5, 0.9],
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length 
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels
        self.model_name = model_name
        self.pretrained_pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=self.device
        )
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=self.device
        )
        self.decision_scores_ = None
        self._fitted = False

    #def _infer_id_and_timestamp(self, df, id_column, timestamp_column):
    #    if id_column not in df.columns:
    #        id_column = None
    #    if timestamp_column not in df.columns:
    #        timestamp_column = None
    #    return id_column, timestamp_column

    #def _convert_dataframe(self, data):
    #    if isinstance(data, np.ndarray):
    #        df = pd.DataFrame(data, columns=[f"target_{i}" for i in range(data.shape[1])])
    #        # If univariate, rename to "target"
    #        if df.shape[1] == 1:
    #            df = df.rename(columns={"target_0": "target"})
    #        return df
    #    return data.copy()
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
        # data shape: (time_steps, num_variates)
        self.score_list = []
        
        data_win, data_target = self.create_dataset(
            data, 
            slidingWindow=self.context_length, 
            predict_time_steps=self.prediction_length
        )
        # data_win shape: (num_windows, num_variates, context_length)
        # data_target shape: (num_windows, num_variates, prediction_length)

        # Prepare inputs for predict_quantiles
        quantiles, mean = self.pretrained_pipeline.predict_quantiles(
            inputs=data_win,
            prediction_length=self.prediction_length,
            quantile_levels=[0.5],
        )
        # mean shape: (num_windows, num_variates, prediction_length)

        # Compute MSE scores across all variates
        predictions = np.array(mean)  # Shape: (num_windows, num_variates, prediction_length)
        
        # MSE per window, averaged across variates
        scores = np.mean((data_target - predictions) ** 2, axis=(1, 2))  # Shape: (num_windows,)
        
        # Pad beginning
        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.context_length + self.prediction_length - 1] = scores[0]
        padded_decision_scores[self.context_length + self.prediction_length - 1:] = scores

        self.decision_scores_ = padded_decision_scores
        return self.decision_scores_

    def fit(self, data):
        # data shape: (time_steps, num_variates)
        self.score_list = []
        
        data_win, data_target = self.create_dataset(
            data, 
            slidingWindow=self.context_length, 
            predict_time_steps=self.prediction_length
        )
        # data_win shape: (num_windows, num_variates, context_length)

        # Prepare training inputs for fine-tuning
        train_inputs = [{"target": data_win[i]} for i in range(data_win.shape[0])]

        # Fine-tune model
        self.pipeline = self.pretrained_pipeline.fit(
            inputs=train_inputs,
            prediction_length=self.prediction_length,
            num_steps=50,
            batch_size=32,
            learning_rate=1e-5,
        )

        # Get predictions from fine-tuned model using numpy API
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=data_win,
            prediction_length=self.prediction_length,
            quantile_levels=[0.5],
        )

        predictions = np.array(mean)  # Shape: (num_windows, num_variates, prediction_length)
        
        # MSE per window, averaged across variates
        scores = np.mean((data_target - predictions) ** 2, axis=(1, 2))
        
        # Pad beginning
        padded_decision_scores = np.zeros(len(data))
        padded_decision_scores[: self.context_length + self.prediction_length - 1] = scores[0]
        padded_decision_scores[self.context_length + self.prediction_length - 1:] = scores

        self.decision_scores_ = padded_decision_scores
        self._fitted = True

    def decision_function(self, X, use_pretrained=False):
        if use_pretrained:
            return self.zero_shot(X)
        else:
            return self.zero_shot(X)
    #def zero_shot(
    #    self,
    #    data: pd.DataFrame,
    #    future_covariates: pd.DataFrame = None,
    #    id_column: str = "_id",
    #    timestamp_column: str = "_timestamp",
    #    use_pretrained: bool = True,
    #):
    #    pipeline_to_use = self.pretrained_pipeline if use_pretrained else self.pipeline
    #    df = self._convert_dataframe(data)
    #    id_column, timestamp_column = self._infer_id_and_timestamp(
    #        df, id_column, timestamp_column
    #    )
#
 #       if id_column is None:
  #          df["_id"] = 0
   #         id_column = "_id"
    #    if timestamp_column is None:
     #       df["_timestamp"] = np.arange(len(df))
      #      timestamp_column = "_timestamp"
#
        #df, id_column = self._ensure_id(df, id_column)
        #df, timestamp_column = self._ensure_timestamp(df, timestamp_column)
 #       target_cols = [c for c in df.columns if c not in {id_column, timestamp_column}]
  #      print(f"[DEBUG] df shape: {df.shape}")
   #     print(f"[DEBUG] target_cols: {target_cols}")
#
 #       pred_df = pipeline_to_use.predict_df(
  #          df=df,
   #         future_df=future_covariates,
    #        prediction_length=self.prediction_length,
     #       quantile_levels=self.quantile_levels,
      #      id_column=id_column,
       #     timestamp_column=timestamp_column,
        #    target=target_cols,
        #)
        #print(f"[DEBUG] pred_df shape: {pred_df.shape}")
        #print(f"[DEBUG] pred_df columns: {pred_df.columns.tolist()}")
        #print(f"[DEBUG] pred_df head:\n{pred_df.head()}")
#
        #scores_list = []
 #       for target in target_cols:
  #          actual = df[target].values
   #         print(f"[DEBUG] target: {target}, actual shape: {actual.shape}")
            #pred = pred_df.loc[
            #    pred_df["target_name"] == target, "predictions"
            #].values
    #        preds = pred_df.loc[pred_df["target_name"] == target, "predictions"]
     #       if preds.empty:
      #          continue
       #     pred = preds.iloc[0]
#
 #           print(f"[DEBUG] pred type: {type(pred)}, pred shape/length: {np.array(pred).shape if hasattr(pred, '__len__') else 'scalar'}")
  #          print(f"[DEBUG] pred value: {pred}")
#
 #           padded_scores = self._compute_scores(actual, pred)
  #          print(f"[DEBUG] padded_scores shape: {padded_scores.shape}")
   #         scores_list.append(padded_scores)
#
 #       self.decision_scores_ = np.mean(np.array(scores_list), axis=0)
  #      return self.decision_scores_
    #def _prepare_inputs(self, data, past_covariates=None, future_covariates=None, id_column="_id", timestamp_column="_timestamp"):
    #    train_inputs = []
    #    for item_id, group in data.groupby(id_column):
    #        # determine target columns per group
    #        tc = [c for c in group.columns if c not in {id_column, timestamp_column}]
    #        if len(tc) == 1:
    #            target = group[tc[0]].values
    #        else:
    #            target = group[tc].values.T#
#
    #        input_dict = {"target": target}
            # Past covariates
    #        if past_covariates:
               # input_dict["past_covariates"] = {col: group[col].values for col in past_covariates}
    #        else:
    #            input_dict["past_covariates"] = {}
#
    #        # Future covariates
    #        if future_covariates:
    #           input_dict["future_covariates"] = {col: None for col in future_covariates}
    #        else:
    #            input_dict["future_covariates"] = {}#
#
    #        train_inputs.append(input_dict)
    #    return train_inputs
#    def fit(
#        self,
#        data: pd.DataFrame,
#        past_covariates=None,
#        future_covariates=None,
#        id_column="_id",
#        timestamp_column="_timestamp",
#        num_steps=1000,
#        batch_size=32,
#        learning_rate=1e-5,
#    ):
#        df = self._convert_dataframe(data)
#        #df, id_column = self._ensure_id(df, id_column)
#        id_column, timestamp_column = self._infer_id_and_timestamp(df, id_column, timestamp_column)
#
#        if id_column is None:
#            df["_id"] = 0
#            id_column = "_id"
#        if timestamp_column is None:
#            df["_timestamp"] = np.arange(len(df))
#            timestamp_column = "_timestamp"
#
#        train_inputs = self._prepare_inputs(
#            df, past_covariates=past_covariates, future_covariates=future_covariates, id_column=id_column, timestamp_column=timestamp_column
#        )
#
#        self.pipeline = self.pipeline.fit(
#            inputs=train_inputs,
#            prediction_length=self.prediction_length,
#            num_steps=num_steps,
#            batch_size=batch_size,
#            learning_rate=learning_rate,
#        )
#        self._fitted = True
#        #self.decision_scores_ = self.zero_shot(
        #    data, future_covariates=future_covariates,
        #    id_column=id_column, timestamp_column=timestamp_column
        #)
#
#    def decision_function(self, data, future_covariates=None, id_column="_id", timestamp_column="_timestamp", use_pretrained=False):
#        if not self._fitted and not use_pretrained:
#            raise RuntimeError("Model must be fine-tuned before calling decision_function")#
#        return self.zero_shot(data, future_covariates, id_column, timestamp_column, use_pretrained=use_pretrained)