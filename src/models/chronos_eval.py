"""Chronos zero-shot forecasting model for HB_NORTH day-ahead prices."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

TARGET = "price_HB_NORTH"
CONTEXT_LEN = 512   # ~3 weeks of hourly history, matches PatchTST
NUM_SAMPLES = 20    # probabilistic samples; median taken as point forecast
BATCH_SIZE = 64     # context windows per predict() call


class ChronosModel:
    """
    Zero-shot 1-step-ahead forecaster using Amazon Chronos (T5-based).

    No training is performed.  fit() loads the pretrained pipeline and
    caches the last context_len rows of training data.  predict() builds
    sliding context windows over the test set and calls the pipeline in
    batches, returning the median of num_samples quantile samples.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        finetuned_path: str | None = None,   # e.g. "models/chronos_finetuned/best"
        context_len: int = CONTEXT_LEN,
        num_samples: int = NUM_SAMPLES,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.finetuned_path = finetuned_path
        self.context_len = context_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self._pipeline = None
        self._tail: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "ChronosModel":
        from chronos import ChronosPipeline  # lazy import — heavy dependency

        load_path = self.finetuned_path if self.finetuned_path else self.model_name
        logger.info("Loading Chronos pipeline: %s", load_path)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        if self.finetuned_path:
            from transformers import T5ForConditionalGeneration
            logger.info("Loading fine-tuned weights from %s", self.finetuned_path)
            finetuned = T5ForConditionalGeneration.from_pretrained(
                self.finetuned_path, torch_dtype=torch.bfloat16
            )
            self._pipeline.model.model.load_state_dict(finetuned.state_dict())
            self._pipeline.model.model.to(next(self._pipeline.model.model.parameters()).device)
        y = train_df[TARGET].values.astype(np.float32)
        self._tail = y[-self.context_len :].copy()
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        y_test = test_df[TARGET].values.astype(np.float32)

        # Prepend training tail so every test step has a full context window
        y_full = np.concatenate([self._tail, y_test])
        n_test = len(y_test)

        # Build (n_test, context_len) array of context windows
        windows = np.stack(
            [y_full[i : i + self.context_len] for i in range(n_test)]
        ).astype(np.float32)

        all_preds: list[np.ndarray] = []
        for start in range(0, n_test, self.batch_size):
            batch = torch.from_numpy(windows[start : start + self.batch_size])
            # forecast: (batch, num_samples, prediction_length=1)
            forecast = self._pipeline.predict(
                batch,
                prediction_length=1,
                num_samples=self.num_samples,
            )
            median = np.median(forecast.numpy(), axis=1)[:, 0]
            all_preds.append(median)

            if start % (self.batch_size * 10) == 0:
                logger.info(
                    "Chronos predict: %d / %d windows done", start, n_test
                )

        preds = np.concatenate(all_preds)
        return pd.Series(preds, index=test_df.index, name=TARGET)
