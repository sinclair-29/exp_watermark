# Prompt Set

This folder contains the default prompt file used by
[`scripts/run_baselines.sh`](/Users/sinclair/Documents/Code/exp_watermark/scripts/run_baselines.sh).

- `prompts.txt` now contains 500 C4-derived continuation prompts.
- The prompts were extracted from the public MarkLLM dataset artifact
  `dataset/c4/processed_c4.json`.
- This repository stores only the normalized `prompt` field, one prompt per line,
  so the batch script can run directly without additional preprocessing.

For reproducible comparisons, keep the same prompt file fixed across all
model/method runs.
