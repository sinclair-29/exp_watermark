import argparse
import json
import math
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Watermarking - Full Implementation")

    parser.add_argument("--model_path", type=str, default="./models/Phi-3-mini-128k-instruct", help="Model path")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")

    parser.add_argument("--gamma", type=float, default=0.5, help="Green list ratio (0-1)")
    parser.add_argument("--delta", type=float, default=2.0, help="KGW logit bias delta")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="OPT watermark threshold beta: when B(p_t, G_t) <= beta, sample only from green list",
    )
    parser.add_argument(
        "--watermark_type",
        type=str,
        default="morph",
        choices=["none", "kgw", "opt", "morph"],
        help="Watermark type: none, KGW, OPT, or MorphMark.",
    )
    parser.add_argument("--hash_window", type=int, default=1, help="Hash window size h")
    parser.add_argument("--hard", action="store_true", help="Use hard red-list masking")
    parser.add_argument("--private_key", type=str, default=None, help="Optional key used by the simple seeded PRF")
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple",
        choices=["simple", "kgw_simple", "private"],
        help="Seeding scheme; private/selfhash mode is intentionally unsupported in this minimal baseline.",
    )

    parser.add_argument(
        "--morph_variant",
        type=str,
        default="exp",
        choices=["linear", "exp", "log"],
        help="MorphMark growth variant: linear / exp / log",
    )
    parser.add_argument("--morph_p0", type=float, default=0.15, help="MorphMark watermarking threshold p0")
    parser.add_argument("--morph_eps", type=float, default=1e-10, help="MorphMark epsilon")
    parser.add_argument("--morph_k_linear", type=float, default=1.55, help="MorphMark k_linear")
    parser.add_argument("--morph_k_exp", type=float, default=1.30, help="MorphMark k_exp")
    parser.add_argument("--morph_k_log", type=float, default=2.15, help="MorphMark k_log")

    parser.set_defaults(do_sample=True)
    parser.add_argument("--do_sample", dest="do_sample", action="store_true", help="Sample tokens during generation")
    parser.add_argument("--no_sample", dest="do_sample", action="store_false", help="Disable sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling threshold")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling threshold")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for HF generation")
    parser.add_argument(
        "--use_beam_search",
        action="store_true",
        help="Compatibility alias that disables sampling and uses beam search",
    )

    parser.add_argument(
        "--ignore_repeated_ngrams",
        action="store_true",
        help="Ignore repeated local n-grams during detection",
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=2,
        help="Legacy text-detection n-gram width; the main token-level detector uses hash_window instead.",
    )
    parser.add_argument("--detection_threshold", type=float, default=2.0, help="Z-score detection threshold")

    parser.add_argument("--compute_ppl", action="store_true", help="Compute real completion-only log-PPL / PPL")
    parser.add_argument("--oracle_model_path", type=str, default=None, help="Optional oracle model for PPL evaluation")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write a single JSON result")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Optional path to append a JSONL result")

    parser.add_argument("--simulate_attack", action="store_true", help="Simulate an attack")
    parser.add_argument("--attack_budget", type=float, default=0.1, help="Attack budget (modification ratio)")
    parser.add_argument(
        "--attack_type",
        type=str,
        default="random",
        choices=["random", "adversarial"],
        help="Attack type",
    )
    return parser


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: str, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


def _append_jsonl(path: str, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(_to_jsonable(payload), sort_keys=True) + "\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_watermarking.analysis import (
        compute_completion_logppl_and_ppl,
        compute_perplexity_bound,
        compute_theoretical_bounds,
        simulate_attack,
    )
    from llm_watermarking.config import WatermarkConfig
    from llm_watermarking.detection import WatermarkDetector
    from llm_watermarking.generation import generate_with_watermark
    from llm_watermarking.watermarking import ensure_supported_seeding_scheme

    normalized_seeding_scheme = ensure_supported_seeding_scheme(args.seeding_scheme)

    config = WatermarkConfig(
        gamma=args.gamma,
        delta=args.delta,
        beta=args.beta,
        watermark_type=args.watermark_type,
        hash_window=args.hash_window,
        seeding_scheme=normalized_seeding_scheme,
        private_key=args.private_key,
        morph_variant=args.morph_variant,
        morph_p0=args.morph_p0,
        morph_eps=args.morph_eps,
        morph_k_linear=args.morph_k_linear,
        morph_k_exp=args.morph_k_exp,
        morph_k_log=args.morph_k_log,
    )

    do_sample = args.do_sample
    num_beams = args.num_beams
    if args.use_beam_search:
        do_sample = False
        if num_beams == 1:
            num_beams = 4

    print("=" * 60)
    print("LLM Watermarking - KGW + OPT + MorphMark integrated implementation")
    print("=" * 60)
    print(f"\nLoading model: {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nGenerating text...")
    generation_result = generate_with_watermark(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        config=config,
        hard=args.hard,
        do_sample=do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=num_beams,
    )

    detector = WatermarkDetector(tokenizer, config)
    detection_result = detector.detect_tokens(
        generation_result["full_ids"],
        prompt_len=generation_result["prompt_len"],
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
    )

    ppl_result = None
    if args.compute_ppl:
        oracle_model = model
        if args.oracle_model_path:
            print(f"\nLoading oracle model for PPL: {args.oracle_model_path}...")
            oracle_model = AutoModelForCausalLM.from_pretrained(
                args.oracle_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        oracle_device = getattr(oracle_model, "device", None)
        input_ids = torch.tensor([generation_result["full_ids"]], dtype=torch.long, device=oracle_device)
        attention_mask = torch.ones_like(input_ids)
        ppl_result = compute_completion_logppl_and_ppl(
            oracle_model,
            input_ids,
            prompt_len=generation_result["prompt_len"],
            attention_mask=attention_mask,
        )

    print(f"\nPrompt:\n{'-' * 40}")
    print(args.prompt)
    print(f"{'-' * 40}")

    print(f"\nGenerated text:\n{'-' * 40}")
    print(generation_result["generated_text"])
    print(f"{'-' * 40}")

    print(f"\nFull text:\n{'-' * 40}")
    print(generation_result["full_text"])
    print(f"{'-' * 40}")

    print(f"\nWatermark type: {config.watermark_type}")
    print(f"Generated length: {generation_result['generated_len']}")
    print(f"Scored generated tokens: {detection_result['num_tokens_scored']}")
    print(f"z-score: {detection_result['z_score']:.2f}")
    print(f"p-value: {detection_result['p_value']:.2e}")
    print(f"Green fraction: {detection_result['green_fraction']:.3f}")

    if detection_result["z_score"] > args.detection_threshold:
        print(f"\n[OK] Watermark detection result: z > {args.detection_threshold}")
    else:
        print(f"\n[X] Watermark detection result: z <= {args.detection_threshold}")

    metadata = generation_result["metadata"]
    if config.watermark_type == "opt":
        print(
            f"\nOPT statistics: applied={metadata.get('opt_applied_tokens', 0)}/"
            f"{metadata.get('num_steps', 0)}, "
            f"fraction={metadata.get('opt_applied_fraction', 0) or 0:.3f}, "
            f"avg_B={metadata.get('avg_opt_B')}"
        )
    elif config.watermark_type == "morph":
        print("\nMorphMark statistics:")
        print(f"  variant: {config.morph_variant}")
        print(
            f"  applied={metadata.get('morph_applied_tokens', 0)}/"
            f"{metadata.get('num_steps', 0)} "
            f"(fraction={metadata.get('morph_applied_fraction', 0) or 0:.3f})"
        )
        avg_p_green = metadata.get("avg_morph_P_G")
        avg_r = metadata.get("avg_morph_r")
        print(f"  avg P_G: {avg_p_green:.4f}" if avg_p_green is not None else "  avg P_G: N/A")
        print(f"  avg r:   {avg_r:.4f}" if avg_r is not None else "  avg r:   N/A")

    if config.watermark_type == "kgw" and metadata.get("avg_spike_entropy") is not None:
        bounds = compute_theoretical_bounds(
            config,
            metadata["avg_spike_entropy"],
            generation_result["generated_len"],
        )
        theoretical_ppl_bound_factor = compute_perplexity_bound(config)
        print("\nTheoretical KGW analysis:")
        print(f"  avg spike entropy: {metadata['avg_spike_entropy']:.4f}")
        print(f"  expected green lower bound: {bounds['expected_green_lower_bound']:.2f}")
        print(f"  variance upper bound: {bounds['variance_upper_bound']:.2f}")
        print(f"  theoretical_ppl_bound_factor: {theoretical_ppl_bound_factor:.3f}")

    if ppl_result is not None:
        print("\nCompletion PPL:")
        print(f"  log_ppl: {ppl_result['log_ppl']:.4f}")
        print(f"  ppl: {ppl_result['ppl']:.4f}" if math.isfinite(ppl_result["ppl"]) else "  ppl: inf")
        print(f"  num_scored_tokens: {ppl_result['num_scored_tokens']}")

    attack_result = None
    if args.simulate_attack:
        print(f"\nSimulating attack (budget epsilon={args.attack_budget}, type={args.attack_type})...")
        attacked_text, attack_info = simulate_attack(
            generation_result["full_text"],
            tokenizer,
            args.attack_budget,
            config,
            args.attack_type,
        )
        attacked_detection = detector.detect(attacked_text, ignore_repeated_ngrams=args.ignore_repeated_ngrams)
        attack_result = {
            "attacked_text": attacked_text,
            "attack_info": attack_info,
            "detection": attacked_detection,
        }

        print(f"\nAttacked text:\n{'-' * 40}")
        print(attacked_text)
        print(f"{'-' * 40}")
        print(f"\nDetection result after attack: z={attacked_detection['z_score']:.2f}, p={attacked_detection['p_value']:.2e}")

    output_payload = {
        "prompt": args.prompt,
        "config": vars(args),
        "generation": generation_result,
        "detection": detection_result,
        "ppl": ppl_result,
        "attack": attack_result,
    }

    if args.output_json:
        _write_json(args.output_json, output_payload)
    if args.output_jsonl:
        _append_jsonl(args.output_jsonl, output_payload)

    print("\n" + "=" * 60)
