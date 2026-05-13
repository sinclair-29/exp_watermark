import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Watermarking - Full Implementation")

    parser.add_argument("--model_path", type=str, default="./models/Phi-3-mini-128k-instruct", help="Model path")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate")

    parser.add_argument("--gamma", type=float, default=0.5, help="Green list ratio (0-1)")
    parser.add_argument("--delta", type=float, default=2.0, help="Logit bias delta, used only for KGW/soft watermark")
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
        choices=["kgw", "opt", "morph"],
        help="Watermark type: 'kgw' is the original additive red/green vocabulary; "
        "'opt' is the OPT method from Optimizing Watermarks; "
        "'morph' is MorphMark adaptive watermarking.",
    )
    parser.add_argument("--hash_window", type=int, default=1, help="Hash window size h")
    parser.add_argument("--hard", action="store_true", help="Use hard red list (Algorithm 1)")
    parser.add_argument("--private_key", type=str, default=None, help="Private watermark key")
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple",
        choices=["simple", "private"],
        help="Seeding scheme: simple (Algorithm 2) or private (Algorithm 3)",
    )

    parser.add_argument(
        "--morph_variant",
        type=str,
        default="exp",
        choices=["linear", "exp", "log"],
        help="MorphMark growth variant: linear / exp / log (MorphMarkexp performs best in the paper)",
    )
    parser.add_argument("--morph_p0", type=float, default=0.15, help="MorphMark watermarking threshold p0 (paper Eq.10)")
    parser.add_argument(
        "--morph_eps",
        type=float,
        default=1e-10,
        help="MorphMark epsilon, a negligibly small positive value",
    )
    parser.add_argument("--morph_k_linear", type=float, default=1.55, help="MorphMark k_linear")
    parser.add_argument("--morph_k_exp", type=float, default=1.30, help="MorphMark k_exp")
    parser.add_argument("--morph_k_log", type=float, default=2.15, help="MorphMark k_log")

    parser.add_argument("--use_beam_search", action="store_true", help="Use Beam Search")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")

    parser.add_argument(
        "--ignore_repeated_ngrams",
        action="store_true",
        help="Ignore repeated n-grams during detection",
    )
    parser.add_argument("--ngram_size", type=int, default=2, help="N-gram size")
    parser.add_argument("--detection_threshold", type=float, default=2.0, help="Z-score detection threshold")

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


def main():
    parser = build_parser()
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_watermarking.analysis import compute_perplexity_bound, compute_theoretical_bounds, simulate_attack
    from llm_watermarking.config import WatermarkConfig
    from llm_watermarking.detection import WatermarkDetector
    from llm_watermarking.generation import generate_with_watermark

    config = WatermarkConfig(
        gamma=args.gamma,
        delta=args.delta,
        beta=args.beta,
        watermark_type=args.watermark_type,
        hash_window=args.hash_window,
        seeding_scheme=args.seeding_scheme,
        private_key=args.private_key,
        morph_variant=args.morph_variant,
        morph_p0=args.morph_p0,
        morph_eps=args.morph_eps,
        morph_k_linear=args.morph_k_linear,
        morph_k_exp=args.morph_k_exp,
        morph_k_log=args.morph_k_log,
    )

    print("=" * 60)
    print("LLM Watermarking - KGW + OPT + MorphMark integrated implementation")
    print("=" * 60)
    print(f"\nConfiguration parameters:")
    print(f"  gamma: {config.gamma}")
    print(f"  Watermark type: {config.watermark_type}")
    print(f"  delta: {config.delta}  # used by KGW")
    print(f"  beta: {config.beta}    # used by OPT")
    if config.watermark_type.lower() == "morph":
        print(f"  MorphMark variant: {config.morph_variant}")
        print(f"  MorphMark p0: {config.morph_p0}")
        print(
            f"  MorphMark k_linear / k_exp / k_log: "
            f"{config.morph_k_linear} / {config.morph_k_exp} / {config.morph_k_log}"
        )
    print(f"  Hash window h: {config.hash_window}")
    if args.hard:
        scheme_desc = "Hard Red List (Alg.1)"
    elif config.watermark_type.lower() == "opt":
        scheme_desc = "OPT Watermark"
    elif config.watermark_type.lower() == "morph":
        scheme_desc = f"MorphMark ({config.morph_variant})"
    else:
        scheme_desc = "Soft Red List / KGW (Alg.2)"
    print(f"  Scheme: {scheme_desc}")
    print(f"  Seeding scheme: {config.seeding_scheme}")
    if args.use_beam_search:
        print(f"  Beam Search: {args.num_beams} beams")

    print(f"\nLoading model: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\nGenerating watermarked text...")
    text, metadata = generate_with_watermark(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        config=config,
        hard=args.hard,
        use_beam_search=args.use_beam_search,
        num_beams=args.num_beams,
    )

    print(f"\nGenerated text:\n{'-' * 40}")
    print(text)
    print(f"{'-' * 40}")
    print(f"\nGeneration metadata: {metadata}")

    print("\nDetecting watermark...")
    detector = WatermarkDetector(tokenizer, config)
    result = detector.detect(
        text,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
        ngram_size=args.ngram_size,
    )

    print(f"\nDetection result:")
    print(f"  z-score: {result['z_score']:.2f}")
    print(f"  p-value: {result['p_value']:.2e}")
    print(f"  Detected tokens: {result['num_tokens']}")
    print(f"  Green tokens: {result['num_green_tokens']}")
    print(
        f"  Green fraction: {result['green_fraction']:.3f} "
        f"(expected: {result['expected_green_fraction']:.3f})"
    )

    if result["z_score"] > args.detection_threshold:
        print(f"\n[OK] Watermark detection result: text is very likely AI-generated (z > {args.detection_threshold})")
    else:
        print(f"\n[X] Watermark detection result: cannot confirm text is AI-generated (z <= {args.detection_threshold})")

    if config.watermark_type.lower() == "opt":
        print(
            f"\nOPT statistics: applied={metadata.get('opt_applied_tokens', 0)}/"
            f"{metadata.get('num_tokens_generated', 0)}, "
            f"fraction={metadata.get('opt_applied_fraction', 0):.3f}, "
            f"avg_B={metadata.get('avg_opt_B')}"
        )
    elif config.watermark_type.lower() == "morph":
        print(f"\nMorphMark statistics:")
        print(f"  variant: {metadata.get('morph_variant')}")
        print(
            f"  applied={metadata.get('morph_applied_tokens', 0)}/"
            f"{metadata.get('num_tokens_generated', 0)} "
            f"(fraction={metadata.get('morph_applied_fraction', 0):.3f})"
        )
        avg_PG = metadata.get("avg_morph_P_G")
        avg_r = metadata.get("avg_morph_r")
        print(f"  avg P_G: {avg_PG:.4f}" if avg_PG is not None else "  avg P_G: N/A")
        print(f"  avg r:   {avg_r:.4f}" if avg_r is not None else "  avg r:   N/A")
    elif "avg_spike_entropy" in metadata and metadata["avg_spike_entropy"] > 0:
        print("\nTheoretical bound analysis (Theorem 4.2):")
        bounds = compute_theoretical_bounds(
            config,
            metadata["avg_spike_entropy"],
            metadata["num_tokens_generated"],
        )
        print(f"  Lower bound on expected green tokens: {bounds['expected_green_lower_bound']:.1f}")
        print(f"  Upper bound on variance: {bounds['variance_upper_bound']:.1f}")
        print(f"  Upper bound on standard deviation: {bounds['std_upper_bound']:.1f}")

    if config.watermark_type.lower() == "kgw":
        ppl_multiplier = compute_perplexity_bound(config)
        print(f"\nPerplexity upper-bound factor (Theorem 4.3): {ppl_multiplier:.3f}x")

    if args.simulate_attack:
        print(f"\nSimulating attack (budget epsilon={args.attack_budget}, type={args.attack_type})...")
        attacked_text, attack_info = simulate_attack(
            text,
            tokenizer,
            args.attack_budget,
            config,
            args.attack_type,
        )

        print(f"\nAttacked text:\n{'-' * 40}")
        print(attacked_text)
        print(f"{'-' * 40}")

        attacked_result = detector.detect(attacked_text)
        print(f"\nDetection result after attack:")
        print(f"  z-score: {attacked_result['z_score']:.2f} (original: {result['z_score']:.2f})")
        print(f"  p-value: {attacked_result['p_value']:.2e}")
        print(f"  Green fraction: {attacked_result['green_fraction']:.3f}")

        if attacked_result["z_score"] > args.detection_threshold:
            print(f"\n[OK] Watermark still detectable after attack")
        else:
            print(f"\n[X] Attack successfully removed the watermark")

    print("\n" + "=" * 60)
