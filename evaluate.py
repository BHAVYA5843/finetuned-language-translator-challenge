"""
Challenge 1 — Evaluation Script
Computes BLEU, chrF, chrF++, TER on three test sets:
  1. flores-devtest  (general domain)
  2. OPUS Ubuntu / GNOME (software domain proxy)
  3. Dataset_Challenge_1.xlsx (proprietary software domain)

Usage:
    python evaluate.py \
        --baseline_model Helsinki-NLP/opus-mt-en-nl \
        --finetuned_model ./outputs/lora_model \
        --custom_dataset ./Dataset_Challenge_1.xlsx \
        --output_dir ./results \
        --batch_size 32
"""

import os
import argparse
import json
import logging
from pathlib import Path

import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def load_model(model_path: str, device: str):
    logger.info(f"Loading model from: {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def translate_batch(model, tokenizer, src_texts: list, device: str,
                    batch_size: int = 32, max_length: int = 256,
                    num_beams: int = 4) -> list:
    """Translate a list of source strings; returns hypotheses."""
    all_hyps = []
    for i in tqdm(range(0, len(src_texts), batch_size), desc="Translating"):
        batch = src_texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        all_hyps.extend(decoded)
    return all_hyps


def compute_metrics(hypotheses: list, references: list) -> dict:
    """
    Compute BLEU, chrF, chrF++, TER using SacreBLEU.
    corpus_score expects refs as [[ref_sent_1, ref_sent_2, ...]]
    (one inner list per reference translation; single-reference = list-of-one-list).
    """
    bleu   = BLEU(effective_order=True)
    chrf   = CHRF()
    chrfpp = CHRF(word_order=2)
    ter    = TER()

    refs = [references]   # single reference; wrap in outer list

    bleu_score   = bleu.corpus_score(hypotheses, refs)
    chrf_score   = chrf.corpus_score(hypotheses, refs)
    chrfpp_score = chrfpp.corpus_score(hypotheses, refs)
    ter_score    = ter.corpus_score(hypotheses, refs)

    return {
        "BLEU":          round(bleu_score.score, 2),
        "chrF":          round(chrf_score.score, 2),
        "chrF++":        round(chrfpp_score.score, 2),
        "TER":           round(ter_score.score, 2),
        "BP":            round(bleu_score.bp, 4),
        "num_sentences": len(hypotheses),
    }


# ─────────────────────────────────────────
# Test Set Loaders
# ─────────────────────────────────────────

def load_flores_devtest(max_samples: int = 1012) -> tuple:
    """
    Load FLORES-200 devtest EN -> NL.
    Primary: facebook/flores (separate dataset per language).
    Fallback: openlanguagedata/flores_plus (all languages, filter by iso code).
    """
    logger.info("Loading FLORES-200 devtest ...")

    try:
        ds_en = load_dataset("facebook/flores", "eng_Latn", split="devtest", trust_remote_code=True)
        ds_nl = load_dataset("facebook/flores", "nld_Latn", split="devtest", trust_remote_code=True)
        src = [ex["sentence"] for ex in ds_en][:max_samples]
        ref = [ex["sentence"] for ex in ds_nl][:max_samples]
        logger.info(f"FLORES devtest loaded (facebook/flores): {len(src)} sentences")
        return src, ref
    except Exception as e:
        logger.warning(f"facebook/flores failed: {e}")

    try:
        ds = load_dataset("openlanguagedata/flores_plus", split="devtest", trust_remote_code=True)
        src = [ex["sentence"] for ex in ds if ex.get("iso_639_3") == "eng"][:max_samples]
        ref = [ex["sentence"] for ex in ds if ex.get("iso_639_3") == "nld"][:max_samples]
        min_len = min(len(src), len(ref))
        src, ref = src[:min_len], ref[:min_len]
        logger.info(f"FLORES devtest loaded (flores_plus): {len(src)} sentences")
        return src, ref
    except Exception as e2:
        raise RuntimeError("Could not load any FLORES devtest variant.") from e2


def load_software_domain_test() -> tuple:
    """
    Load a software-domain EN-NL test set.
    Priority: OPUS Ubuntu -> OPUS GNOME.
    """
    logger.info("Loading software domain test set ...")

    for name, dataset_id in [("OPUS Ubuntu", "opus_ubuntu"), ("OPUS GNOME", "opus_gnome")]:
        try:
            ds = load_dataset(dataset_id, "en-nl", split="train", trust_remote_code=True)
            ds = ds.shuffle(seed=99).select(range(min(2000, len(ds))))
            src = [ex["translation"]["en"] for ex in ds]
            ref = [ex["translation"]["nl"] for ex in ds]
            logger.info(f"{name} loaded: {len(src)} sentences")
            return src, ref
        except Exception as e:
            logger.warning(f"{name} unavailable: {e}")

    raise RuntimeError(
        "No software domain test set could be loaded (tried opus_ubuntu, opus_gnome). "
        "Use --skip_wmt to skip."
    )


def load_custom_dataset(xlsx_path: str) -> tuple:
    """Load the challenge proprietary EN->NL dataset from xlsx."""
    logger.info(f"Loading custom dataset from: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]

    # Auto-detect source and reference columns
    src_col, ref_col = None, None
    for c in df.columns:
        cl = c.lower()
        if src_col is None and ("english" in cl or cl == "source"):
            src_col = c
    for c in df.columns:
        cl = c.lower()
        if ref_col is None and c != src_col and (
            "reference" in cl or "dutch" in cl or "nl" in cl or
            ("translation" in cl and c != src_col)
        ):
            ref_col = c

    if src_col is None or ref_col is None:
        raise ValueError(
            f"Could not auto-detect source/reference columns. Found: {list(df.columns)}"
        )

    logger.info(f"  Source col: '{src_col}'  |  Reference col: '{ref_col}'")
    df = df[[src_col, ref_col]].dropna()
    src = df[src_col].astype(str).tolist()
    ref = df[ref_col].astype(str).tolist()
    logger.info(f"Custom dataset: {len(src)} sentence pairs")
    return src, ref


# ─────────────────────────────────────────
# Per-sentence analysis
# ─────────────────────────────────────────

def per_sentence_bleu(hypotheses: list, references: list) -> list:
    bleu = BLEU(effective_order=True)
    return [round(bleu.sentence_score(h, [r]).score, 2) for h, r in zip(hypotheses, references)]


def save_detailed_results(hypotheses, references, sources, scores_per_sent, out_path: str):
    df = pd.DataFrame({
        "source":     sources,
        "reference":  references,
        "hypothesis": hypotheses,
        "sent_BLEU":  scores_per_sent,
    })
    df.to_excel(out_path, index=False)
    logger.info(f"Per-sentence results saved to {out_path}")


# ─────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────

def evaluate_model(model, tokenizer, label: str, test_sets: dict,
                   device: str, batch_size: int, output_dir: Path) -> dict:
    all_results = {}
    for ds_name, (src, ref) in test_sets.items():
        logger.info(f"  [{label}] Evaluating on: {ds_name}")
        hyps = translate_batch(model, tokenizer, src, device, batch_size)
        metrics = compute_metrics(hyps, ref)
        all_results[ds_name] = metrics
        logger.info(f"  [{label}] {ds_name}: {metrics}")

        if ds_name == "Custom (Challenge 1)":
            sent_bleu = per_sentence_bleu(hyps, ref)
            slug = label.replace(" ", "_").replace("(", "").replace(")", "")
            save_detailed_results(
                hyps, ref, src, sent_bleu,
                str(output_dir / f"{slug}_custom_detailed.xlsx")
            )
            (output_dir / f"{slug}_custom_hyps.txt").write_text(
                "\n".join(hyps), encoding="utf-8"
            )
    return all_results


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned MarianMT EN->NL")
    parser.add_argument("--baseline_model",  default="Helsinki-NLP/opus-mt-en-nl")
    parser.add_argument("--finetuned_model", default="./outputs/finetuned_model")
    parser.add_argument("--custom_dataset",  default="./Dataset_Challenge_1.xlsx")
    parser.add_argument("--output_dir",      default="./results")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--num_beams",       type=int, default=4)
    parser.add_argument("--skip_flores",     action="store_true",
                        help="Skip FLORES devtest (faster iteration)")
    parser.add_argument("--skip_wmt",        action="store_true",
                        help="Skip OPUS software domain test set")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Assemble test sets ──────────────────────────────────
    test_sets = {}

    if not args.skip_flores:
        try:
            test_sets["FLORES devtest (general)"] = load_flores_devtest()
        except Exception as e:
            logger.error(f"Skipping FLORES: {e}")

    if not args.skip_wmt:
        try:
            test_sets["WMT/OPUS (software domain)"] = load_software_domain_test()
        except Exception as e:
            logger.error(f"Skipping software domain set: {e}")

    if os.path.exists(args.custom_dataset):
        try:
            test_sets["Custom (Challenge 1)"] = load_custom_dataset(args.custom_dataset)
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
    else:
        logger.warning(f"Custom dataset not found at: {args.custom_dataset}")

    if not test_sets:
        raise RuntimeError("No test sets could be loaded. Aborting.")

    # ── Baseline ────────────────────────────────────────────
    baseline_model, baseline_tok = load_model(args.baseline_model, device)
    baseline_results = evaluate_model(
        baseline_model, baseline_tok,
        "Baseline (pretrained)", test_sets, device, args.batch_size, output_dir
    )
    del baseline_model
    torch.cuda.empty_cache()

    # ── Fine-tuned ──────────────────────────────────────────
    ft_results = {}
    if os.path.exists(args.finetuned_model):
        ft_model, ft_tok = load_model(args.finetuned_model, device)
        ft_results = evaluate_model(
            ft_model, ft_tok,
            "Fine-tuned", test_sets, device, args.batch_size, output_dir
        )
        del ft_model
        torch.cuda.empty_cache()
    else:
        logger.warning(
            f"Fine-tuned model not found at '{args.finetuned_model}'. "
            "Only baseline results will be saved."
        )

    # ── Results table ───────────────────────────────────────
    rows = []
    for ds_name in test_sets:
        for metric in ["BLEU", "chrF", "chrF++", "TER"]:
            row = {
                "Dataset":    ds_name,
                "Metric":     metric,
                "Baseline":   baseline_results.get(ds_name, {}).get(metric, "N/A"),
                "Fine-tuned": ft_results.get(ds_name, {}).get(metric, "N/A"),
            }
            b, f = row["Baseline"], row["Fine-tuned"]
            if isinstance(b, float) and isinstance(f, float):
                row["Delta"] = f"{f - b:+.2f}"
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps({"baseline": baseline_results, "finetuned": ft_results}, indent=2)
    )

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\n" + results_df.to_string(index=False))
    logger.info(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
