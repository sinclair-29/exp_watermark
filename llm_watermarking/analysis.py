import math
from typing import Dict, Optional, Tuple

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
    Theoretical KGW perplexity upper-bound factor from the paper.

    This is not actual perplexity and should be reported separately from real PPL.
    """
    alpha = np.exp(config.delta)
    return 1 + (alpha - 1) * config.gamma


def compute_completion_logppl_and_ppl(
    model,
    input_ids,
    prompt_len: int,
    attention_mask: Optional[object] = None,
) -> Dict:
    """
    Compute real completion-only log-PPL / PPL.

    ``labels[:, :prompt_len] = -100`` is correct for Hugging Face causal LMs
    because the implementation shifts labels internally. The first generated token
    at position ``prompt_len`` is therefore predicted from the previous prompt
    token and still counted, while the prompt tokens themselves are excluded.
    """
    import torch

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)

    prompt_len = max(0, min(int(prompt_len), input_ids.size(1)))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # HF causal LMs shift labels internally, so labels[:, 1:] is the set of
    # targets that actually contribute to the loss.
    num_scored_tokens = int((labels[:, 1:] != -100).sum().item())
    if num_scored_tokens == 0:
        return {
            "log_ppl": float("nan"),
            "mean_nll": float("nan"),
            "ppl": float("inf"),
            "num_scored_tokens": 0,
            "total_nll": 0.0,
        }

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    mean_nll = float(outputs.loss.detach().cpu().item())
    try:
        ppl = math.exp(mean_nll)
    except OverflowError:
        ppl = float("inf")

    return {
        "log_ppl": mean_nll,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "num_scored_tokens": num_scored_tokens,
        "total_nll": mean_nll * num_scored_tokens,
    }


def compute_completion_ppl_from_text(
    model,
    tokenizer,
    prompt: str,
    completion: str,
) -> Dict:
    """
    Convenience helper for text inputs.

    The main CLI uses generated token ids directly and should prefer
    ``compute_completion_logppl_and_ppl`` to avoid decode+retokenize mismatch.
    """
    import torch

    device = getattr(model, "device", None)
    prompt_batch = tokenizer(prompt, return_tensors="pt")
    completion_batch = tokenizer(completion, return_tensors="pt", add_special_tokens=False)

    prompt_ids = prompt_batch.input_ids
    completion_ids = completion_batch.input_ids
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

    attention_mask = getattr(prompt_batch, "attention_mask", None)
    completion_attention_mask = getattr(completion_batch, "attention_mask", None)
    if attention_mask is not None and completion_attention_mask is not None:
        attention_mask = torch.cat([attention_mask, completion_attention_mask], dim=1)

    if device is not None:
        full_ids = full_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    return compute_completion_logppl_and_ppl(
        model,
        full_ids,
        prompt_len=prompt_ids.size(1),
        attention_mask=attention_mask,
    )


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
