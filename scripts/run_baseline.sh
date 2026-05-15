#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available in PATH." >&2
  exit 1
fi

PROMPT_FILE="${1:-data/prompts.txt}"
BEGIN_INDEX="${2:-1}"
END_INDEX="${3:-}"
FORCE_VALUE="${FORCE:-0}"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Error: prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ ! "$BEGIN_INDEX" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: BEGIN_INDEX must be a positive integer: $BEGIN_INDEX" >&2
  exit 1
fi

if [[ -n "$END_INDEX" && ! "$END_INDEX" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: END_INDEX must be a positive integer: $END_INDEX" >&2
  exit 1
fi

if [[ -n "$END_INDEX" ]] && (( BEGIN_INDEX > END_INDEX )); then
  echo "Error: BEGIN_INDEX ($BEGIN_INDEX) cannot be greater than END_INDEX ($END_INDEX)." >&2
  exit 1
fi

MODEL_PHI3="../LLMJailbreak/models/Phi-3-mini-128k-instruct"
MODEL_QWEN="../LLMJailbreak/models/Qwen2.5-7B-Instruct"
MODEL_LLAMA="../LLMJailbreak/models/Llama-2-7b-chat-hf"

declare -A MODEL_PATHS=(
  ["Phi-3-mini"]="$MODEL_PHI3"
  ["Qwen2.5-7B"]="$MODEL_QWEN"
  ["Llama-2-7B"]="$MODEL_LLAMA"
)

for model_name in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[$model_name]}"
  if [[ ! -d "$model_path" ]]; then
    echo "Error: model directory not found for $model_name: $model_path" >&2
    exit 1
  fi
done

RESULTS_DIR="results/baseline"
RAW_DIR="$RESULTS_DIR/raw"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
SUMMARY_MD="$RESULTS_DIR/summary.md"

mkdir -p "$RAW_DIR"

MAX_NEW_TOKENS=50
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=0
NUM_BEAMS=1

METHOD_KEYS=(
  "none"
  "kgw"
  "opt"
  "morph_linear"
  "morph_exp"
  "morph_log"
)

method_display_name() {
  case "$1" in
    none) echo "No Watermark" ;;
    kgw) echo "KGW" ;;
    opt) echo "OPT" ;;
    morph_linear|morph_exp|morph_log) echo "MorphMark" ;;
    *) echo "Unknown" ;;
  esac
}

method_variant_name() {
  case "$1" in
    morph_linear) echo "linear" ;;
    morph_exp) echo "exp" ;;
    morph_log) echo "log" ;;
    *) echo "-" ;;
  esac
}

method_cli_args() {
  case "$1" in
    none)
      printf '%s\n' --watermark_type none
      ;;
    kgw)
      printf '%s\n' --watermark_type kgw --gamma 0.5 --delta 2.0
      ;;
    opt)
      printf '%s\n' --watermark_type opt --gamma 0.5 --beta 0.0
      ;;
    morph_linear)
      printf '%s\n' --watermark_type morph --morph_variant linear
      ;;
    morph_exp)
      printf '%s\n' --watermark_type morph --morph_variant exp
      ;;
    morph_log)
      printf '%s\n' --watermark_type morph --morph_variant log
      ;;
    *)
      echo "Error: unsupported method key: $1" >&2
      exit 1
      ;;
  esac
}

prompt_index=0
selected_count=0
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "${prompt//[[:space:]]/}" ]]; then
    continue
  fi

  prompt_index=$((prompt_index + 1))

  if (( prompt_index < BEGIN_INDEX )); then
    continue
  fi

  if [[ -n "$END_INDEX" ]] && (( prompt_index > END_INDEX )); then
    break
  fi

  selected_count=$((selected_count + 1))
  prompt_id="$(printf 'prompt_%04d' "$prompt_index")"

  for model_name in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$model_name]}"

    for method_key in "${METHOD_KEYS[@]}"; do
      method_name="$(method_display_name "$method_key")"
      variant_name="$(method_variant_name "$method_key")"
      condition_dir="$RAW_DIR/$model_name/$method_key"
      output_json="$condition_dir/${prompt_id}.json"

      mkdir -p "$condition_dir"

      if [[ "$FORCE_VALUE" != "1" && -s "$output_json" ]]; then
        echo "Skipping existing result: model=$model_name method=$method_name variant=$variant_name prompt=$prompt_id"
        continue
      fi

      mapfile -t method_args < <(method_cli_args "$method_key")

      echo "Running: model=$model_name method=$method_name variant=$variant_name prompt=$prompt_id"

      python -m llm_watermarking \
        --model_path "$model_path" \
        --prompt "$prompt" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --do_sample \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --num_beams "$NUM_BEAMS" \
        --compute_ppl \
        --output_json "$output_json" \
        "${method_args[@]}"
    done
  done
done < "$PROMPT_FILE"

if (( prompt_index == 0 )); then
  echo "Error: no non-empty prompts found in $PROMPT_FILE" >&2
  exit 1
fi

if (( selected_count == 0 )); then
  if [[ -n "$END_INDEX" ]]; then
    echo "Error: selected prompt range [$BEGIN_INDEX, $END_INDEX] contains no non-empty prompts." >&2
  else
    echo "Error: selected prompt range starting at $BEGIN_INDEX contains no non-empty prompts." >&2
  fi
  exit 1
fi

python scripts/summarize_baseline_results.py \
  --raw_dir "$RAW_DIR" \
  --summary_csv "$SUMMARY_CSV" \
  --summary_md "$SUMMARY_MD"
