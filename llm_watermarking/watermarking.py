import hashlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from llm_watermarking.config import WatermarkConfig


def hash_tokens(tokens: List[int], key: str = "", scheme: str = "simple") -> int:
    """
    Convert multiple tokens into a pseudo-random seed.

    Args:
        tokens: list of tokens (tokens within the window)
        key: optional private key
        scheme: hashing scheme ("simple", "private")

    Returns:
        Pseudo-random seed
    """
    hash_input = "_".join(str(token) for token in tokens)
    if key:
        hash_input = f"{key}_{hash_input}"

    h = hashlib.sha256(hash_input.encode()).hexdigest()
    return int(h, 16) % (2**32)


def partition_vocab(vocab_size: int, seed: int, gamma: float = 0.5) -> Tuple[Set[int], Set[int]]:
    """
    Partition the vocabulary into a green list and a red list.

    Paper Algorithm 2 Step 3:
    Using this random number generator, randomly partition the vocabulary
    into a "green list" G of size gamma|V|, and a "red list" R of size (1-gamma)|V|.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(vocab_size)
    split = int(gamma * vocab_size)
    green = set(perm[:split].tolist())
    red = set(perm[split:].tolist())
    return green, red


def compute_spike_entropy(probs, z_modulus: float) -> float:
    """
    Compute Spike Entropy (Paper Definition 4.1).

    S(p, z) = sum_k p_k / (1 + z * p_k)

    Spike entropy measures the concentration of the distribution, used to analyze
    the relationship between watermark strength and text entropy.
    """
    probs_np = probs.detach().cpu().numpy()
    spike_entropy = np.sum(probs_np / (1 + z_modulus * probs_np))
    return float(spike_entropy)


def _get_simple_seed(prev_tokens: List[int], hash_window: int, private_key: str = "") -> int:
    window = prev_tokens[-hash_window:]
    return hash_tokens(window, key=private_key)


def _get_private_selfhash_seed(
    prev_tokens: List[int],
    current_token: int,
    hash_window: int,
    private_key: str = "",
) -> int:
    min_hash = float("inf")
    best_seed = 0

    for prev_token in prev_tokens[-hash_window:]:
        hashed = hash_tokens([current_token, prev_token], key=private_key, scheme="private")
        if hashed < min_hash:
            min_hash = hashed
            best_seed = hashed

    return best_seed


def _apply_kgw_bias(logits, green_indices, delta: float):
    import torch

    bias = torch.zeros_like(logits)
    bias[green_indices] = delta
    return logits + bias


def _compute_opt_damage_B(logits, green: Set[int], eps: float = 1e-12) -> Tuple[float, float]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    green_indices = torch.tensor(list(green), device=logits.device, dtype=torch.long)
    green_mask = torch.zeros_like(probs, dtype=torch.bool)
    green_mask[green_indices] = True

    gamma_t = probs[green_mask].sum().clamp(eps, 1.0 - eps)
    indicator = green_mask.to(probs.dtype)
    coeff = (gamma_t - indicator) / (gamma_t * (1.0 - gamma_t))
    b_value = torch.sum(coeff * probs * torch.log(probs.clamp_min(eps)))
    return float(b_value.detach().cpu()), float(gamma_t.detach().cpu())


def _compute_morph_phi(config: WatermarkConfig, p_green: float) -> float:
    p0 = config.morph_p0
    eps_val = config.morph_eps

    if p_green <= p0:
        return eps_val

    variant = (config.morph_variant or "exp").lower()
    if variant == "linear":
        z = config.morph_k_linear * p_green
    elif variant == "exp":
        z = float(np.exp(config.morph_k_exp * p_green) - 1.0)
    elif variant == "log":
        z = float(np.log(config.morph_k_log * p_green + 1.0))
    else:
        z = config.morph_k_linear * p_green

    return min(max(z, eps_val), 1.0 - eps_val)


def _apply_morphmark(
    logits,
    green: Set[int],
    config: WatermarkConfig,
    eps: float = 1e-20,
) -> Tuple[object, Dict]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)

    green_indices = torch.tensor(list(green), device=logits.device, dtype=torch.long)
    green_mask = torch.zeros_like(probs, dtype=torch.bool)
    green_mask[green_indices] = True

    p_green_t = probs[green_mask].sum()
    p_green = float(p_green_t.clamp(min=1e-12, max=1.0 - 1e-12).detach().cpu())
    r = _compute_morph_phi(config, p_green)

    green_factor = 1.0 + r * (1.0 - p_green) / p_green
    red_factor = 1.0 - r

    new_probs = torch.where(
        green_mask,
        probs * green_factor,
        probs * red_factor,
    )
    new_probs = new_probs / new_probs.sum().clamp_min(eps)
    new_logits = torch.log(new_probs.clamp_min(eps))

    info = {
        "mode": "morph",
        "morph_variant": config.morph_variant,
        "P_G": p_green,
        "r": r,
        "applied": r > config.morph_eps * 10,
    }
    return new_logits, info


class WatermarkLogitsProcessor:
    def __init__(self, config: WatermarkConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        self.green_list_cache = {}

    def _get_seed(self, prev_tokens: List[int], current_token: Optional[int] = None) -> int:
        """Get random seed according to the configured scheme."""
        if self.config.seeding_scheme == "private" and current_token is not None:
            return _get_private_selfhash_seed(
                prev_tokens,
                current_token,
                self.config.hash_window,
                self.config.private_key or "",
            )

        return _get_simple_seed(
            prev_tokens,
            self.config.hash_window,
            self.config.private_key or "",
        )

    def _get_green_red_list(self, seed: int) -> Tuple[Set[int], Set[int]]:
        """Get or cache the green/red list."""
        if seed not in self.green_list_cache:
            self.green_list_cache[seed] = partition_vocab(
                self.vocab_size,
                seed,
                self.config.gamma,
            )
        return self.green_list_cache[seed]

    def __call__(
        self,
        logits,
        prev_tokens: List[int],
        hard: bool = False,
        return_info: bool = False,
    ):
        import torch

        info = {"watermark_type": self.config.watermark_type, "applied": False}

        if len(prev_tokens) < 1:
            return (logits, info) if return_info else logits

        seed = self._get_seed(prev_tokens)
        green, _ = self._get_green_red_list(seed)
        green_indices = torch.tensor(list(green), device=logits.device, dtype=torch.long)

        if hard:
            mask = torch.full_like(logits, float("-inf"))
            mask[green_indices] = 0
            logits = logits + mask
            info.update({"applied": True, "mode": "hard"})

        elif self.config.watermark_type.lower() == "opt":
            b_value, gamma_t = _compute_opt_damage_B(logits, green)
            info.update({"mode": "opt", "B": b_value, "Gamma_t": gamma_t, "beta": self.config.beta})

            if b_value <= self.config.beta:
                mask = torch.full_like(logits, float("-inf"))
                mask[green_indices] = 0
                logits = logits + mask
                info["applied"] = True

        elif self.config.watermark_type.lower() == "morph":
            logits, morph_info = _apply_morphmark(logits, green, self.config)
            info.update(morph_info)

        else:
            logits = _apply_kgw_bias(logits, green_indices, self.config.delta)
            info.update({"applied": True, "mode": "kgw"})

        return (logits, info) if return_info else logits


class PrivateWatermarkLogitsProcessor(WatermarkLogitsProcessor):
    def __call__(self, logits, prev_tokens: List[int], hard: bool = False) -> Tuple[object, Optional[int]]:
        import torch

        if len(prev_tokens) < 1:
            return logits, None

        sorted_indices = torch.argsort(logits, descending=True)
        top_logit = logits[sorted_indices[0]].item()

        for index in range(len(sorted_indices)):
            candidate_token = sorted_indices[index].item()
            candidate_logit = logits[sorted_indices[index]].item()

            if candidate_logit < top_logit - self.config.delta:
                return logits, sorted_indices[0].item()

            seed = self._get_seed(prev_tokens, current_token=candidate_token)
            green, _ = self._get_green_red_list(seed)

            rng = np.random.default_rng(seed)
            is_green = rng.random() < self.config.gamma

            if is_green:
                return logits, candidate_token

        return logits, sorted_indices[0].item()


def hash_token(token, seed=0):
    """Hash function compatible with the original interface."""
    return hash_tokens([token], key=str(seed) if seed else "")


def watermark_sampling(logits, prev_token, gamma=0.5, delta=2.0, hard=False):
    """KGW watermark sampling function compatible with the original interface."""
    config = WatermarkConfig(gamma=gamma, delta=delta, watermark_type="kgw")
    processor = WatermarkLogitsProcessor(config, logits.shape[-1])
    return processor(logits, [prev_token], hard=hard)


def opt_watermark_sampling(logits, prev_token, gamma=0.5, beta=0.0):
    """Compatibility interface for the OPT method from the Optimizing Watermarks paper."""
    config = WatermarkConfig(gamma=gamma, beta=beta, watermark_type="opt")
    processor = WatermarkLogitsProcessor(config, logits.shape[-1])
    return processor(logits, [prev_token], hard=False)


def morph_watermark_sampling(
    logits,
    prev_token,
    gamma: float = 0.5,
    variant: str = "exp",
    p0: float = 0.15,
    k_linear: float = 1.55,
    k_exp: float = 1.30,
    k_log: float = 2.15,
):
    """Compatibility interface for MorphMark."""
    config = WatermarkConfig(
        gamma=gamma,
        watermark_type="morph",
        morph_variant=variant,
        morph_p0=p0,
        morph_k_linear=k_linear,
        morph_k_exp=k_exp,
        morph_k_log=k_log,
    )
    processor = WatermarkLogitsProcessor(config, logits.shape[-1])
    return processor(logits, [prev_token], hard=False)
