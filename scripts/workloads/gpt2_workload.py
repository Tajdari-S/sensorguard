#!/usr/bin/env python3
"""P0 GPT-2 adapter: --mode train (wikitext LM fine-tuning) or --mode infer
(prefill+decode generation, the matched inference control)."""

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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t) > 200]

    end = raw_now() + args.duration_s
    t0 = raw_now()

    if args.mode == "train":
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        tokens = 0
        loss_val = float("nan")
        i = 0
        while raw_now() < end:
            batch_texts = [texts[(i + j) % len(texts)] for j in range(args.batch_size)]
            i += args.batch_size
            enc = tok(batch_texts, return_tensors="pt", truncation=True,
                      max_length=args.seq_len, padding="max_length").to(device)
            out = model(**enc, labels=enc["input_ids"])
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            tokens += int(enc["input_ids"].numel())
            loss_val = float(out.loss.detach())
        torch.cuda.synchronize(device)
        useful = {"mode": "train_gpt2_wikitext", "tokens": tokens,
                  "tokens_per_s": round(tokens / (raw_now() - t0), 1),
                  "final_loss": round(loss_val, 4)}
    else:
        model.eval()
        generated = 0
        prompts = 0
        i = 0
        with torch.inference_mode():
            while raw_now() < end:
                batch_texts = [texts[(i + j) % len(texts)][:400] for j in range(args.batch_size)]
                i += args.batch_size
                enc = tok(batch_texts, return_tensors="pt", truncation=True,
                          max_length=args.seq_len // 2, padding=True).to(device)
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
                generated += int(out.numel() - enc["input_ids"].numel())
                prompts += args.batch_size
        torch.cuda.synchronize(device)
        useful = {"mode": "infer_gpt2", "prompts": prompts, "new_tokens": generated,
                  "tokens_per_s": round(generated / (raw_now() - t0), 1)}

    print("useful_work " + json.dumps(useful))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
