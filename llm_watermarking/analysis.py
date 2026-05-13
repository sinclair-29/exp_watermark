from typing import Dict, Tuple

import numpy as np

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import hash_tokens, partition_vocab


def compute_theoretical_bounds(
    config: WatermarkConfig,
    avg_spike_entropy: float,
    num_tokens: int,
) -> Dict:
    alpha = np.exp(config.delta)
    gamma = config.gamma

    expected_green_lower = (gamma * alpha * num_tokens * avg_spike_entropy) / (
        1 + (alpha - 1) * gamma
    )

    if gamma >= 0.5:
        variance_upper = num_tokens * gamma * (1 - gamma)
    else:
        p_green = (gamma * alpha * avg_spike_entropy) / (1 + (alpha - 1) * gamma)
        variance_upper = num_tokens * p_green * (1 - p_green)

    if abs(gamma - 0.5) < 0.01 and abs(config.delta - np.log(2)) < 0.1:
        expected_simplified = (2 / 3) * num_tokens * avg_spike_entropy
        variance_simplified = (2 / 3) * num_tokens * avg_spike_entropy * (
            1 - (2 / 3) * avg_spike_entropy
        )
    else:
        expected_simplified = None
        variance_simplified = None

    return {
        "expected_green_lower_bound": expected_green_lower,
        "variance_upper_bound": variance_upper,
        "std_upper_bound": np.sqrt(variance_upper),
        "expected_simplified": expected_simplified,
        "variance_simplified": variance_simplified,
    }


def compute_perplexity_bound(config: WatermarkConfig) -> float:
    """
    Compute perplexity upper bound (Paper Theorem 4.3).

    E[perplexity] <= (1 + (alpha-1)*gamma) * P*

    where P* is the perplexity of the original model.
    """
    alpha = np.exp(config.delta)
    return 1 + (alpha - 1) * config.gamma


def simulate_attack(
    text: str,
    tokenizer,
    attack_budget: float,
    config: WatermarkConfig,
    attack_type: str = "random",
) -> Tuple[str, Dict]:
    tokens = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    num_modifications = int(len(tokens) * attack_budget)

    if attack_type == "random":
        modify_positions = np.random.choice(
            len(tokens),
            size=min(num_modifications, len(tokens)),
            replace=False,
        )

        for pos in modify_positions:
            tokens[pos] = np.random.randint(0, tokenizer.vocab_size)

    elif attack_type == "adversarial":
        green_positions = []
        for i in range(config.hash_window, len(tokens)):
            prev_tokens = tokens[:i]
            seed = hash_tokens(prev_tokens[-config.hash_window :], key=config.private_key or "")
            green, red = partition_vocab(tokenizer.vocab_size, seed, config.gamma)
            if tokens[i] in green:
                green_positions.append((i, list(red)))

        np.random.shuffle(green_positions)
        for pos, red_list in green_positions[:num_modifications]:
            if red_list:
                tokens[pos] = np.random.choice(red_list)

    attacked_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return attacked_text, {
        "attack_type": attack_type,
        "budget": attack_budget,
        "num_modifications": num_modifications,
        "original_length": len(tokens),
    }
