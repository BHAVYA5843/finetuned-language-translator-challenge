"""
train_lora.py
=============
LoRA fine-tuning for Helsinki-NLP/opus-mt-en-nl using PEFT + PyTorch Lightning.

Why LoRA is better here than full fine-tuning:
  - Only trains ~0.5% of parameters → preserves pretrained EN-NL knowledge
  - Much less catastrophic forgetting (which caused the v1 regression)
  - Better generalisation from small datasets

LoRA config targets the attention Q/V projection matrices in both encoder
and decoder, which are the most important for translation quality.

Usage:

    # Step 1 — LoRA fine-tuning
    python train_lora.py \
        --dataset_dir     ./custom_dataset \
        --output_dir      ./outputs/lora_model \
        --max_epochs      10 \
        --batch_size      16 \
        --learning_rate   3e-4

    # Step 2 — evaluate
    python evaluate.py \
        --finetuned_model ./outputs/lora_model \
        --custom_dataset  ./Dataset_Challenge_1.xlsx \
        --output_dir      ./results_lora
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, src_texts: list, tgt_texts: list, tokenizer, max_length: int = 128):
        self.src_texts  = src_texts
        self.tgt_texts  = tgt_texts
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        model_inputs = self.tokenizer(
            self.src_texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=self.tgt_texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, dataset_dir: str, batch_size: int, max_length: int):
        super().__init__()
        self.tokenizer   = tokenizer
        self.dataset_dir = Path(dataset_dir)
        self.batch_size  = batch_size
        self.max_length  = max_length

    def setup(self, stage=None):
        train_df = pd.read_csv(self.dataset_dir / "train.csv").dropna()
        val_df   = pd.read_csv(self.dataset_dir / "val.csv").dropna()
        logger.info(f"Train: {len(train_df)}  Val: {len(val_df)}")

        self.train_ds = TranslationDataset(
            train_df["src"].tolist(), train_df["ref"].tolist(),
            self.tokenizer, self.max_length
        )
        self.val_ds = TranslationDataset(
            val_df["src"].tolist(), val_df["ref"].tolist(),
            self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)


# ─────────────────────────────────────────
# LoRA Model
# ─────────────────────────────────────────

def build_lora_model(model_name: str, lora_r: int, lora_alpha: int,
                     lora_dropout: float) -> MarianMTModel:
    """
    Wrap MarianMT with LoRA adapters on Q and V projections.

    Target modules: MarianMT uses 'q_proj' and 'v_proj' in its attention layers.
    We apply LoRA to both encoder and decoder attention.
    """
    base_model = MarianMTModel.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,
        r              = lora_r,
        lora_alpha     = lora_alpha,
        lora_dropout   = lora_dropout,
        # Target Q and V projections in both encoder and decoder self-attention
        # and decoder cross-attention
        target_modules = ["q_proj", "v_proj"],
        bias           = "none",
        inference_mode = False,
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


class LoRAFineTuner(pl.LightningModule):

    MODEL_NAME = "Helsinki-NLP/opus-mt-en-nl"

    def __init__(self, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, learning_rate: float = 3e-4,
                 warmup_steps: int = 50, weight_decay: float = 0.01,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = build_lora_model(
            self.MODEL_NAME, lora_r, lora_alpha, lora_dropout
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out.loss, on_step=True, on_epoch=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("val_loss", out.loss, on_epoch=True, prog_bar=True)
        return out.loss

    def configure_optimizers(self):
        from transformers import get_linear_schedule_with_warmup

        # Only optimize LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def save_merged_model(self, output_dir: str):
        """
        Merge LoRA weights back into the base model and save as a standard
        HuggingFace model. This means evaluate.py works without any PEFT
        dependency at inference time.
        """
        logger.info("Merging LoRA weights into base model...")
        merged = self.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged model saved to: {output_dir}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune opus-mt-en-nl")

    # Data
    p.add_argument("--dataset_dir",   default="./custom_dataset",
                   help="Directory with train.csv and val.csv from build_combined_dataset.py")

    # Output
    p.add_argument("--output_dir",    default="./outputs/lora_model")
    p.add_argument("--log_dir",       default="./outputs/tb_logs_lora")

    # LoRA
    p.add_argument("--lora_r",        type=int,   default=16,
                   help="LoRA rank. Higher = more capacity but more params. 8–32 is typical.")
    p.add_argument("--lora_alpha",    type=int,   default=32,
                   help="LoRA scaling factor. Usually 2x lora_r.")
    p.add_argument("--lora_dropout",  type=float, default=0.1)

    # Training
    p.add_argument("--max_epochs",    type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--learning_rate", type=float, default=3e-4,
                   help="LoRA uses higher LR than full fine-tuning (3e-4 to 1e-3)")
    p.add_argument("--max_length",    type=int,   default=128)
    p.add_argument("--warmup_steps",  type=int,   default=50)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--precision",     default="16-mixed")
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    lit_model = LoRAFineTuner(
        lora_r         = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        learning_rate  = args.learning_rate,
        warmup_steps   = args.warmup_steps,
        weight_decay   = args.weight_decay,
        label_smoothing= args.label_smoothing,
    )

    dm = DataModule(
        tokenizer   = lit_model.tokenizer,
        dataset_dir = args.dataset_dir,
        batch_size  = args.batch_size,
        max_length  = args.max_length,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath    = os.path.join(args.output_dir, "checkpoints"),
        filename   = "best-{epoch:02d}-{val_loss:.4f}",
        monitor    = "val_loss",
        mode       = "min",
        save_top_k = 2,
        save_last  = True,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    tb_logger     = TensorBoardLogger(args.log_dir, name="lora-en-nl")

    trainer = pl.Trainer(
        max_epochs              = args.max_epochs,
        precision               = args.precision,
        gradient_clip_val       = args.gradient_clip_val,
        accumulate_grad_batches = args.accumulate_grad_batches,
        callbacks               = [checkpoint_cb, early_stop_cb],
        logger                  = tb_logger,
        log_every_n_steps       = 5,
        val_check_interval      = 1.0,   # validate once per epoch (small dataset)
    )

    logger.info("Starting LoRA training...")
    trainer.fit(lit_model, datamodule=dm)

    # Merge and save as standard HuggingFace model
    best = LoRAFineTuner.load_from_checkpoint(checkpoint_cb.best_model_path)
    best.save_merged_model(args.output_dir)
    logger.info("Done. Run evaluate.py to compare baseline vs LoRA model.")


if __name__ == "__main__":
    main()
