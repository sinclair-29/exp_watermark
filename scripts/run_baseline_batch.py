#!/usr/bin/env python3

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


MODEL_PATHS = {
    "Phi-3-mini": str((ROOT_DIR / "../LLMJailbreak/models/Phi-3-mini-128k-instruct").resolve()),
    "Qwen2.5-7B": str((ROOT_DIR / "../LLMJailbreak/models/Qwen2.5-7B-Instruct").resolve()),
    "Llama-2-7B": str((ROOT_DIR / "../LLMJailbreak/models/Llama-2-7b-chat-hf").resolve()),
}

METHOD_KEYS = (
    "none",
    "kgw",
    "opt",
    "morph_linear",
    "morph_exp",
    "morph_log",
)

RESULTS_DIR = ROOT_DIR / "results" / "baseline"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
SUMMARY_MD = RESULTS_DIR / "summary.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline watermark experiments with one loaded model at a time.")
    parser.add_argument("prompt_file", nargs="?", default="data/prompts.txt", help="Prompt file path")
    parser.add_argument("begin_index", nargs="?", type=int, default=1, help="1-based inclusive prompt start index")
    parser.add_argument("end_index", nargs="?", type=int, default=None, help="1-based inclusive prompt end index")
    parser.add_argument("--models", type=str, default=",".join(MODEL_PATHS.keys()), help="Comma-separated model names")
    parser.add_argument("--methods", type=str, default=",".join(METHOD_KEYS), help="Comma-separated method keys")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed for deterministic per-run generation")
    parser.add_argument("--debug_tokens", action="store_true", help="Store compact per-token diagnostics in raw JSON")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw JSON files")
    return parser


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_args(args: argparse.Namespace) -> Tuple[Path, List[str], List[str], bool]:
    prompt_file = Path(args.prompt_file).expanduser()
    if not prompt_file.is_absolute():
        prompt_file = (Path.cwd() / prompt_file).resolve()
    else:
        prompt_file = prompt_file.resolve()
    if not prompt_file.is_file():
        raise SystemExit(f"Error: prompt file not found: {prompt_file}")

    if args.begin_index < 1:
        raise SystemExit(f"Error: begin_index must be >= 1, got {args.begin_index}")

    if args.end_index is not None and args.end_index < args.begin_index:
        raise SystemExit(
            f"Error: end_index ({args.end_index}) must be >= begin_index ({args.begin_index})"
        )

    selected_models = parse_csv_arg(args.models)
    if not selected_models:
        raise SystemExit("Error: --models must specify at least one model name")
    unsupported_models = [model for model in selected_models if model not in MODEL_PATHS]
    if unsupported_models:
        raise SystemExit(f"Error: unsupported model name(s): {', '.join(unsupported_models)}")

    selected_methods = parse_csv_arg(args.methods)
    if not selected_methods:
        raise SystemExit("Error: --methods must specify at least one method key")
    unsupported_methods = [method for method in selected_methods if method not in METHOD_KEYS]
    if unsupported_methods:
        raise SystemExit(f"Error: unsupported method key(s): {', '.join(unsupported_methods)}")

    for model_name in selected_models:
        model_path = Path(MODEL_PATHS[model_name])
        if not model_path.is_dir():
            raise SystemExit(f"Error: model directory not found for {model_name}: {model_path}")

    force = args.force or os.environ.get("FORCE") == "1"
    return prompt_file, selected_models, selected_methods, force


def load_selected_prompts(prompt_file: Path, begin_index: int, end_index: int | None) -> List[Tuple[int, str, str]]:
    selected: List[Tuple[int, str, str]] = []
    prompt_index = 0

    with prompt_file.open() as handle:
        for line in handle:
            prompt = line.strip()
            if not prompt:
                continue

            prompt_index += 1
            if prompt_index < begin_index:
                continue
            if end_index is not None and prompt_index > end_index:
                break

            prompt_id = f"prompt_{prompt_index:04d}"
            selected.append((prompt_index, prompt_id, prompt))

    if prompt_index == 0:
        raise SystemExit(f"Error: no non-empty prompts found in {prompt_file}")

    if not selected:
        if end_index is not None:
            raise SystemExit(
                f"Error: selected prompt range [{begin_index}, {end_index}] contains no non-empty prompts."
            )
        raise SystemExit(
            f"Error: selected prompt range starting at {begin_index} contains no non-empty prompts."
        )

    return selected


def build_config(method_key: str):
    from llm_watermarking.config import WatermarkConfig

    if method_key == "none":
        return WatermarkConfig(watermark_type="none")
    if method_key == "kgw":
        return WatermarkConfig(watermark_type="kgw", gamma=0.5, delta=2.0)
    if method_key == "opt":
        return WatermarkConfig(watermark_type="opt", gamma=0.5, beta=0.0)
    if method_key == "morph_linear":
        return WatermarkConfig(watermark_type="morph", morph_variant="linear")
    if method_key == "morph_exp":
        return WatermarkConfig(watermark_type="morph", morph_variant="exp")
    if method_key == "morph_log":
        return WatermarkConfig(watermark_type="morph", morph_variant="log")
    raise ValueError(f"Unsupported method key: {method_key}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            return None
    return None


def derive_run_seed(base_seed: int, model_name: str, prompt_index: int, method_key: str) -> int:
    model_index = list(MODEL_PATHS.keys()).index(model_name)
    method_index = METHOD_KEYS.index(method_key)
    seed = int(base_seed) + model_index * 1_000_003 + int(prompt_index) * 10_007 + method_index * 101
    return seed % (2**32)


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_token_diagnostics(detection_result: dict, generation_metadata: dict):
    token_details = detection_result.pop("token_details", None)
    if token_details is None:
        return None

    step_metadata = generation_metadata.get("step_metadata") or []
    diagnostics = []
    for item in token_details:
        diagnostic = {
            "token_index": item.get("token_index"),
            "token_id": item.get("token_id"),
            "seed": item.get("seed"),
            "watermark_vocab_size": item.get("watermark_vocab_size"),
            "is_green": item.get("is_green"),
            "cumulative_green_count": item.get("cumulative_green_count"),
        }

        token_index = item.get("token_index")
        if isinstance(token_index, int) and 0 <= token_index < len(step_metadata):
            batch = step_metadata[token_index].get("batch") or []
            if batch:
                generation_info = batch[0]
                if "seed" in generation_info:
                    diagnostic["generation_seed"] = generation_info["seed"]
                if "P_G" in generation_info:
                    diagnostic["P_G"] = generation_info["P_G"]
                if "r" in generation_info:
                    diagnostic["r"] = generation_info["r"]

        diagnostics.append(diagnostic)

    return diagnostics


def run_for_model(
    model_name: str,
    model_path: str,
    prompts: Sequence[Tuple[int, str, str]],
    selected_methods: Sequence[str],
    args: argparse.Namespace,
    force: bool,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_watermarking.analysis import compute_completion_logppl_and_ppl
    from llm_watermarking.detection import WatermarkDetector
    from llm_watermarking.generation import generate_with_watermark

    print(f"Loading model: {model_name} from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        for prompt_index, prompt_id, prompt in prompts:
            print(f"Prompt {prompt_id}")
            for method_key in selected_methods:
                output_json = RAW_DIR / model_name / method_key / f"{prompt_id}.json"
                if not force and output_json.exists() and output_json.stat().st_size > 0:
                    print(f"Skipping existing result: {output_json}")
                    continue

                run_seed = derive_run_seed(args.seed, model_name, prompt_index, method_key)
                print(f"  Method {method_key} seed={run_seed} -> {output_json}")
                config = build_config(method_key)
                set_seed(run_seed)
                generation_result = generate_with_watermark(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    config=config,
                    hard=False,
                    do_sample=not args.no_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_beams=args.num_beams,
                )

                metadata = generation_result["metadata"]
                detector = WatermarkDetector(tokenizer, config, watermark_vocab_size=metadata.get("watermark_vocab_size"))
                detection_result = detector.detect_tokens(
                    generation_result["full_ids"],
                    prompt_len=generation_result["prompt_len"],
                    ignore_repeated_ngrams=False,
                    return_details=args.debug_tokens,
                )
                token_diagnostics = (
                    compact_token_diagnostics(detection_result, metadata) if args.debug_tokens else None
                )

                model_device = get_model_device(model)
                input_ids = torch.tensor(
                    [generation_result["full_ids"]],
                    dtype=torch.long,
                    device=model_device,
                )
                attention_mask = torch.ones_like(input_ids)
                ppl_result = compute_completion_logppl_and_ppl(
                    model,
                    input_ids,
                    prompt_len=generation_result["prompt_len"],
                    attention_mask=attention_mask,
                )

                payload = {
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "model_name": model_name,
                    "model_path": model_path,
                    "method_key": method_key,
                    "seed": run_seed,
                    "config": vars(config),
                    "generation": generation_result,
                    "detection": detection_result,
                    "ppl": ppl_result,
                }
                if token_diagnostics is not None:
                    payload["token_diagnostics"] = token_diagnostics
                write_json(output_json, payload)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_summary(selected_models: Sequence[str], selected_methods: Sequence[str]) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "summarize_baseline_results.py"),
        "--raw_dir",
        str(RAW_DIR),
        "--summary_csv",
        str(SUMMARY_CSV),
        "--summary_md",
        str(SUMMARY_MD),
        "--models",
        ",".join(selected_models),
        "--methods",
        ",".join(selected_methods),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    prompt_file, selected_models, selected_methods, force = validate_args(args)

    prompts = load_selected_prompts(prompt_file, args.begin_index, args.end_index)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    range_label = (
        f"[{args.begin_index}, {args.end_index}]"
        if args.end_index is not None
        else f"[{args.begin_index}, end]"
    )
    print(f"Selected prompt range: {range_label}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Selected methods: {', '.join(selected_methods)}")
    print(f"Selected non-empty prompts: {len(prompts)}")

    for model_name in selected_models:
        run_for_model(
            model_name=model_name,
            model_path=MODEL_PATHS[model_name],
            prompts=prompts,
            selected_methods=selected_methods,
            args=args,
            force=force,
        )

    run_summary(selected_models, selected_methods)


if __name__ == "__main__":
    main()
