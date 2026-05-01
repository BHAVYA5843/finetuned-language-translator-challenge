"""
Challenge 1 — Quick Inference Script
Translate a single sentence or a plain-text file using either
the pretrained baseline or the fine-tuned model.

Usage:
    # Single sentence
    python translate.py --text "Select a default resolution for this display."

    # Fine-tuned model
    !python scripts/translate.py \
    --model ./outputs/lora_model_v2 \
    --text "Snap crystal-clear photos and selfies in any light with a 64MP external camera and OIS—or a 32MP internal camera with Quad Pixel technology."

    #Baseline model
    !python scripts/translate.py \
    --text "Snap crystal-clear photos and selfies in any light with a 64MP external camera and OIS—or a 32MP internal camera with Quad Pixel technology."

"""

import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer


def translate(model, tokenizer, texts: list[str], device: str,
              num_beams: int = 4, max_length: int = 256) -> list[str]:
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=num_beams,
                                    max_length=max_length, early_stopping=True)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="Helsinki-NLP/opus-mt-en-nl",
                        help="HuggingFace model name or local path")
    parser.add_argument("--text",        default=None, help="Single sentence to translate")
    parser.add_argument("--num_beams",   type=int, default=4)
    parser.add_argument("--max_length",  type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[translate] Device: {device}")
    print(f"[translate] Model: {args.model}")

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device)
    model.eval()

    if args.text:
        result = translate(model, tokenizer, [args.text], device,
                           num_beams=args.num_beams, max_length=args.max_length)
        print(f"\nSource:      {args.text}")
        print(f"Translation: {result[0]}")

    else:
        print("Provide --text ")


if __name__ == "__main__":
    main()
