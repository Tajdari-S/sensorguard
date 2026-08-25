#!/usr/bin/env python3
"""P0 BERT adapter: --mode train (masked-LM fine-tuning on wikitext) or
--mode infer (matched forward-only MLM inference)."""

import argparse
import json
import time

import torch


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-s", type=float, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--mask-prob", type=float, default=0.15)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t) > 200]

    def batch(i):
        chunk = [texts[(i + j) % len(texts)] for j in range(args.batch_size)]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=args.seq_len, padding="max_length")
        labels = enc["input_ids"].clone()
        mask = (torch.rand_like(labels, dtype=torch.float) < args.mask_prob) \
            & (labels != tok.pad_token_id) & (labels != tok.cls_token_id) \
            & (labels != tok.sep_token_id)
        enc["input_ids"] = enc["input_ids"].masked_fill(mask, tok.mask_token_id)
        labels[~mask] = -100
        return {k: v.to(device) for k, v in enc.items()}, labels.to(device)

    end = raw_now() + args.duration_s
    t0 = raw_now()
    tokens = 0
    i = 0

    if args.mode == "train":
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        loss_val = float("nan")
        while raw_now() < end:
            enc, labels = batch(i)
            i += args.batch_size
            out = model(**enc, labels=labels)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            tokens += int(labels.numel())
            loss_val = float(out.loss.detach())
        torch.cuda.synchronize(device)
        useful = {"mode": "train_bert_mlm", "tokens": tokens,
                  "tokens_per_s": round(tokens / (raw_now() - t0), 1),
                  "final_loss": round(loss_val, 4)}
    else:
        model.eval()
        with torch.inference_mode():
            while raw_now() < end:
                enc, _ = batch(i)
                i += args.batch_size
                model(**enc)
                tokens += int(enc["input_ids"].numel())
        torch.cuda.synchronize(device)
        useful = {"mode": "infer_bert_mlm", "tokens": tokens,
                  "tokens_per_s": round(tokens / (raw_now() - t0), 1)}

    print("useful_work " + json.dumps(useful))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
