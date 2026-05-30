"""Fine-tune amazon/chronos-t5-small on ERCOT day-ahead prices."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.pipeline import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_PATH  = "data/processed/ercot_processed.parquet"
BASE_MODEL      = "amazon/chronos-t5-small"
RESUME_FROM     = Path("models/chronos_finetuned/best")   # set to None for fresh start
OUTPUT_DIR      = Path("models/chronos_finetuned_v2")
TARGET          = "price_HB_NORTH"

CONTEXT_LEN     = 512
PRED_LEN        = 64   # must match model's fixed prediction_length; inference still uses 1-step
BATCH_SIZE      = 32
NUM_STEPS       = 2000
LEARNING_RATE   = 5e-6   # halved for second pass
WARMUP_STEPS    = 200
GRAD_CLIP       = 1.0
VAL_EVERY       = 200   # evaluate on val set every N steps


# ── Dataset ───────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """Each sample: (context_len,) prices → (1,) next price."""

    def __init__(self, prices: np.ndarray, context_len: int = CONTEXT_LEN) -> None:
        self.prices = torch.from_numpy(prices.astype(np.float32))
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.prices) - self.context_len - PRED_LEN + 1

    def __getitem__(self, idx: int):
        context = self.prices[idx : idx + self.context_len]
        target  = self.prices[idx + self.context_len : idx + self.context_len + PRED_LEN]
        return context, target


# ── Training ──────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def evaluate(model, tokenizer, loader, device) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for context, target in loader:
            # Tokenize on CPU (tokenizer boundaries live on CPU), then move to device
            input_ids, attention_mask, scale = tokenizer.context_input_transform(context)
            label_ids, _ = tokenizer.label_input_transform(target, scale)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_ids      = label_ids.to(device)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label_ids,
            ).loss
            total_loss += loss.item() * len(context)
            total_n    += len(context)
    model.train()
    return total_loss / total_n


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading %s", PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)
    train_df, val_df, test_df = make_splits(df)

    exog_cols = [c for c in test_df.columns if not c.startswith("price_")]
    test_df = test_df.copy()
    test_df[exog_cols] = test_df[exog_cols].ffill()

    train_prices = train_df[TARGET].values
    val_prices   = np.concatenate([train_df[TARGET].values[-CONTEXT_LEN:],
                                   val_df[TARGET].values])

    train_ds = SlidingWindowDataset(train_prices, CONTEXT_LEN)
    val_ds   = SlidingWindowDataset(val_prices,   CONTEXT_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    logger.info(
        "train windows=%d  val windows=%d", len(train_ds), len(val_ds)
    )

    # ── Load pipeline (resume from checkpoint if set) ────────────────────────
    from chronos import ChronosPipeline
    from transformers import T5ForConditionalGeneration
    logger.info("Loading base pipeline: %s", BASE_MODEL)
    pipeline  = ChronosPipeline.from_pretrained(BASE_MODEL, device_map="auto", dtype=torch.bfloat16)
    t5_model  = pipeline.model.model      # T5ForConditionalGeneration
    tokenizer = pipeline.tokenizer        # MeanScaleUniformBins
    device    = next(t5_model.parameters()).device
    if RESUME_FROM is not None:
        logger.info("Resuming from checkpoint: %s", RESUME_FROM)
        ckpt = T5ForConditionalGeneration.from_pretrained(RESUME_FROM, torch_dtype=torch.bfloat16)
        t5_model.load_state_dict(ckpt.state_dict())
        t5_model.to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(t5_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # ── Training loop ─────────────────────────────────────────────────────────
    t5_model.train()
    step          = 0
    best_val_loss = float("inf")
    t0            = time.time()
    train_iter    = iter(train_loader)

    logger.info("Starting fine-tuning for %d steps (batch=%d, lr=%.0e)", NUM_STEPS, BATCH_SIZE, LEARNING_RATE)

    while step < NUM_STEPS:
        try:
            context, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            context, target = next(train_iter)

        # Tokenize on CPU (tokenizer boundaries live on CPU), then move to device
        input_ids, attention_mask, scale = tokenizer.context_input_transform(context)
        label_ids, _ = tokenizer.label_input_transform(target, scale)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids      = label_ids.to(device)

        # Forward + backward
        lr = get_lr(step, WARMUP_STEPS, NUM_STEPS, LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        loss = t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        ).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(t5_model.parameters(), GRAD_CLIP)
        optimizer.step()
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            logger.info("step %4d/%d  loss=%.4f  lr=%.2e  [%.0fs]",
                        step, NUM_STEPS, loss.item(), lr, elapsed)

        # Validation
        if step % VAL_EVERY == 0 or step == NUM_STEPS:
            val_loss = evaluate(t5_model, tokenizer, val_loader, device)
            logger.info("step %4d  val_loss=%.4f%s",
                        step, val_loss,
                        "  *** best ***" if val_loss < best_val_loss else "")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                t5_model.save_pretrained(OUTPUT_DIR / "best")
                pipeline.tokenizer  # tokenizer is stateless, no save needed
                logger.info("Saved best checkpoint → %s", OUTPUT_DIR / "best")

    # Save final checkpoint
    t5_model.save_pretrained(OUTPUT_DIR / "final")
    logger.info("Saved final checkpoint → %s", OUTPUT_DIR / "final")
    logger.info("Fine-tuning complete in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
