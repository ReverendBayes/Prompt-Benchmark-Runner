#!/usr/bin/env python3
"""
promptbench.py: Prompt-Benchmark Runner

Usage:
    python promptbench.py \
      --prompts prompts.json \
      --test-cases cases.csv \
      --model gpt-4-turbo \
      --output results.csv

Features:
    - Loads prompt templates and test cases (CSV with 'id','input','reference' columns)
    - Runs each prompt-case combination against OpenAI ChatCompletion
    - Records latency, token usage, BLEU and ROUGE-L scores
    - Pluggable scorer interface
Dependencies:
    openai, pandas, nltk, numpy
"""
import os
import argparse
import json
import time
import openai
import pandas as pd
import numpy as np

# -------------------- Scoring Functions --------------------

def bleu_score(pred, ref):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    pred_tokens = pred.split()
    ref_tokens = [ref.split()]
    smoothie = SmoothingFunction().method4
    try:
        return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    except ZeroDivisionError:
        return 0.0

def rouge_l_score(pred, ref):
    # token-level LCS F1
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    m, n = len(ref_tokens), len(pred_tokens)
    # build LCS table
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    return 2 * prec * rec / (prec + rec)

SCORERS = {
    "bleu": bleu_score,
    "rouge_l": rouge_l_score
}

# -------------------- Main Logic --------------------

def main():
    parser = argparse.ArgumentParser(description="Prompt-Benchmark Runner")
    parser.add_argument("--prompts", required=True, help="JSON file with prompt templates")
    parser.add_argument("--test-cases", required=True, help="CSV file with columns: id,input,reference")
    parser.add_argument("--model", default="gpt-4-turbo", help="OpenAI model for ChatCompletion")
    parser.add_argument("--output", required=True, help="CSV output path for results")
    parser.add_argument("--metrics", nargs="+", choices=list(SCORERS.keys()), default=["bleu","rouge_l"],
                        help="List of metrics to compute")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    # Set API key
    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key required. Use --api-key or set OPENAI_API_KEY.")

    # Load prompts and test cases
    with open(args.prompts) as f:
        prompts = json.load(f)
    cases = pd.read_csv(args.test_cases)

    records = []
    for pi, prompt_tpl in enumerate(prompts, start=1):
        for _, row in cases.iterrows():
            case_id = row.get("id", "")
            inp = row["input"]
            ref = row["reference"]
            # Format prompt
            if "{input}" in prompt_tpl:
                prompt = prompt_tpl.replace("{input}", inp)
            else:
                prompt = f"{prompt_tpl}\nInput: {inp}"
            # Call API
            start = time.time()
            resp = openai.ChatCompletion.create(
                model=args.model,
                messages=[{"role":"user","content":prompt}]
            )
            latency = time.time() - start
            content = resp.choices[0].message.content.strip()
            usage = resp.usage or {}
            prompt_tokens = usage.get("prompt_tokens", None)
            completion_tokens = usage.get("completion_tokens", None)
            total_tokens = usage.get("total_tokens", None)
            # Compute metrics
            scores = {}
            for m in args.metrics:
                scores[m] = SCORERS[m](content, ref)
            # Record
            records.append({
                "prompt_id": pi,
                "prompt_template": prompt_tpl,
                "case_id": case_id,
                "input": inp,
                "reference": ref,
                "prediction": content,
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                **scores
            })

    # Save results
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
