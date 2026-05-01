Domain-Specific Machine Translation (EN → NL)
Overview

This project focuses on domain-specific fine-tuning of a small encoder-decoder Transformer model for English to Dutch translation, specifically targeting software/UI text.

The goal is to improve translation quality for:

UI strings
product descriptions
device/system messages

while preserving general translation capability.

1. Model Selection
Base Model

I selected:

Helsinki-NLP MarianMT (opus-mt-en-nl)
Why this model?
Lightweight and efficient → suitable for experimentation
Strong pretrained performance on EN–NL
Encoder-decoder architecture → ideal for translation tasks
Trade-off
Smaller than modern models like NLLB → slightly lower ceiling
But faster and easier to fine-tune in constrained environments

2. Why LoRA instead of Full Fine-Tuning
I used LoRA (Low-Rank Adaptation) implemented via PEFT.

Reference:

Motivation
Problem with full fine-tuning
Overfits quickly on small datasets
Causes catastrophic forgetting
Degrades general-domain performance
Why LoRA?
Trains only a small subset (~0.5%) of parameters
Preserves pretrained knowledge
More stable for low-resource domain adaptation
Faster and memory-efficient
Implementation choice

LoRA was applied to:

q_proj, v_proj (attention layers)

Because:

Attention layers control translation alignment
Most impactful for translation quality
Trade-off
Approach	Pros	Cons
Full fine-tuning	High capacity	Overfitting, unstable
LoRA	Stable, efficient	Slightly lower max capacity

3. Dataset Strategy
Key Decision

I did NOT directly use large unfiltered datasets like:

OPUS (full)
WMT16 (full)
Why not?

1. Domain mismatch
These datasets contain:

news
parliament text
subtitles

which do not match:

software/UI domain

2. Noise and irrelevance
Unfiltered datasets introduce:

irrelevant vocabulary
inconsistent tone
poor domain alignment

3. Negative impact observed
During experimentation:

adding large generic data → reduced BLEU
translations became less UI-specific
Final approach
-Domain-focused dataset
curated UI/software strings
short structured sentences
domain-specific terminology
-Filtered OPUS (limited use)
only UI-relevant samples
strict filtering applied
Important note (data leakage prevention)

The evaluation dataset:

Dataset_Challenge_1.xlsx

was NOT used for training, and only used for final evaluation.

4. Training Pipeline
Implemented using:

PyTorch Lightning
HuggingFace Transformers
PEFT (LoRA)

Reference:

Key configurations
Batch size: 16
Learning rate: 3e-4
Epochs: 10
Label smoothing: 0.1
Gradient accumulation: 2
Training strategy
Early stopping based on validation loss
Checkpoint saving
Mixed precision training

5. Evaluation Strategy
Reference:

Evaluation performed on:

1. General Domain
FLORES devtest

2. Software Domain
OPUS Ubuntu / GNOME (proxy dataset)

3. Challenge Dataset
Dataset_Challenge_1.xlsx
Metrics used
BLEU
chrF
chrF++
TER

6. Results

Reference:

Custom (Challenge Dataset)
Metric	Baseline	Fine-tuned	Delta
BLEU	30.31	52.92	+22.61
chrF	62.88	73.70	+10.82
chrF++	59.51	71.84	+12.33
TER	52.20	33.40	-18.80
Interpretation
Significant improvement in BLEU and chrF indicates better alignment with domain-specific translations
TER reduction shows fewer edits required → improved usability
Qualitative Observations
Improvements
Better grammatical structure
Improved fluency
More consistent terminology
Remaining issues
Some literal translations persist
Occasional incorrect localization
Mixed usage of English and Dutch terms

7. Key Learnings
Domain-specific data is more important than dataset size
LoRA is highly effective for low-resource adaptation
Data quality > data quantity
Overuse of generic corpora degrades domain performance

8. Future Improvements
Better terminology control
Error-driven retraining
Use of stronger base models (e.g., NLLB)
Human-in-the-loop correction

9. How to Run
Training
python train_lora.py --dataset_dir ./custom_dataset --output_dir ./outputs/lora_model

Evaluation
python evaluate.py --finetuned_model ./outputs/lora_model

Inference
python translate.py \
    --model ./outputs/lora_model \
    --text "Snap crystal-clear photos and selfies in any light with a 64MP external camera and OIS—or a 32MP internal camera with Quad Pixel technology."

Final note
This project demonstrates that targeted domain adaptation + efficient fine-tuning (LoRA) can significantly improve translation quality in specialized domains.