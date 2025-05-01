# Prompt-Benchmark Runner
## What it is 
A CLI that loads a list of prompt templates, runs them against your OpenAI (or any) LLM, and spits out a CSV of latency, token-usage, and qualitative scores (via a reference answer).

## Why you need it
* Prompt engineering teams need to A/B test prompts at scale, but current scripts are scattered across notebooks and custom dashboards.
* One file, one command → immediate insights into which prompt earns you the best ROUGE/F1/latency trade-off.

## Key features
* Supports automatic metric computation (BLEU, ROUGE)
* Pluggable scorer interface

**Usage example**:

```bash
python promptbench.py \
  --prompts prompts.json \
  --test-cases cases.csv \
  --model gpt-4-turbo \
  --output results.csv
```

**Dependencies**:

```bash
pip install openai pandas nltk numpy
```
