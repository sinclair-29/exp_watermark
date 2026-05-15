import hashlib
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from llm_watermarking.config import WatermarkConfig

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - exercised only in envs without transformers.
    class LogitsProcessor:  # type: ignore[override]
        """Fallback base class so the package can import without transformers."""

        pass


SUPPORTED_WATERMARK_TYPES = {"none", "kgw", "opt", "morph"}
SUPPORTED_SEEDING_SCHEMES = {"simple", "kgw_simple"}


def hash_tokens(tokens: List[int], key: str = "", scheme: str = "simple") -> int:
    """
    Convert multiple tokens into a pseudo-random seed.

    Args:
        tokens: list of tokens within the current hash window
        key: optional key string used to key the PRF
        scheme: hashing scheme label kept for compatibility

    Returns:
        Pseudo-random seed
    """
    del scheme
    hash_input = "_".join(str(token) for token in tokens)
    if key:
        hash_input = f"{key}_{hash_input}"

    hashed = hashlib.sha256(hash_input.encode()).hexdigest()
    return int(hashed, 16) % (2**32)


def normalize_seeding_scheme(seeding_scheme: str) -> str:
    """Normalize supported seeding aliases."""
    normalized = (seeding_scheme or "simple").lower()
    if normalized == "kgw_simple":
        return "simple"
    return normalized


def ensure_supported_seeding_scheme(seeding_scheme: str) -> str:
    """
    Validate the minimal seeding schemes supported by this repository.

    The private/self-hash mode is intentionally rejected because this small
    baseline does not implement matching generation and detection.
    """
    normalized = normalize_seeding_scheme(seeding_scheme)
    if normalized == "private":
        raise NotImplementedError(
            "private/selfhash mode is not implemented in this minimal baseline; "
            "use seeding_scheme='simple'."
        )
    if normalized not in SUPPORTED_SEEDING_SCHEMES:
        raise ValueError(f"Unsupported seeding scheme: {seeding_scheme}")
    return normalized


def build_greenlist_ids(vocab_size: int, seed: int, gamma: float, device=None):
    """
    Build a deterministic green list using a CPU torch.Generator.

    This intentionally keeps the RNG on CPU so generation and detection can share
    the same green-list logic without CPU/GPU RNG drift.
    """
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**64 - 1))
    permutation = torch.randperm(vocab_size, generator=generator, device="cpu")
    greenlist_size = max(0, min(vocab_size, int(gamma * vocab_size)))
    green_ids = permutation[:greenlist_size]
    if device is not None:
        green_ids = green_ids.to(device)
    return green_ids


def build_greenlist_mask(vocab_size: int, green_ids, device=None):
    """Build a boolean mask from green token ids."""
    import torch

    mask_device = device if device is not None else green_ids.device
    green_mask = torch.zeros(vocab_size, device=mask_device, dtype=torch.bool)
    if green_ids.numel() > 0:
        green_mask[green_ids.to(mask_device)] = True
    return green_mask


def partition_vocab(vocab_size: int, seed: int, gamma: float = 0.5) -> Tuple[Set[int], Set[int]]:
    """
    Compatibility helper returning Python sets.

    Internally the main implementation now uses Torch ids/masks for closer
    alignment with Hugging Face generation code.
    """
    green_ids = build_greenlist_ids(vocab_size, seed, gamma, device="cpu")
    green = set(green_ids.tolist())
    red = set(range(vocab_size)) - green
    return green, red


def compute_spike_entropy(probs, z_modulus: float) -> float:
    """
    Compute Spike Entropy (KGW paper definition).

    S(p, z) = sum_k p_k / (1 + z * p_k)
    """
    probs_np = probs.detach().cpu().numpy()
    spike_entropy = np.sum(probs_np / (1 + z_modulus * probs_np))
    return float(spike_entropy)


def _get_simple_seed(prev_tokens: Sequence[int], hash_window: int, private_key: str = "") -> int:
    context_tokens = list(prev_tokens[-hash_window:]) if hash_window > 0 else []
    return hash_tokens(context_tokens, key=private_key)


def _get_private_selfhash_seed(
    prev_tokens: Sequence[int],
    current_token: int,
    hash_window: int,
    private_key: str = "",
) -> int:
    """
    Compatibility stub kept so the old symbol still exists.

    The minimal baseline intentionally does not use self-hash/private mode.
    """
    raise NotImplementedError(
        "private/selfhash mode is not implemented in this minimal baseline; "
        "use seeding_scheme='simple'."
    )


def _apply_kgw_bias(logits, green_mask, delta: float):
    """Apply the KGW additive bias to green logits only."""
    import torch

    bias = torch.zeros_like(logits)
    bias[green_mask] = delta
    return logits + bias


def _compute_opt_damage_B(logits, green_mask, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Compute the OPT damage criterion B(p_t, G_t) from the paper.
    """
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    probs = probs.clamp_min(eps)
    gamma_t = probs[green_mask].sum().clamp(eps, 1.0 - eps)
    indicator = green_mask.to(probs.dtype)
    coeff = (gamma_t - indicator) / (gamma_t * (1.0 - gamma_t))
    b_value = torch.sum(coeff * probs * torch.log(probs))
    return float(b_value.detach().cpu()), float(gamma_t.detach().cpu())


def _compute_morph_phi(config: WatermarkConfig, p_green: float) -> float:
    if p_green <= config.morph_p0:
        return config.morph_eps

    variant = (config.morph_variant or "exp").lower()
    if variant == "linear":
        z_value = config.morph_k_linear * p_green
    elif variant == "exp":
        z_value = math.exp(config.morph_k_exp * p_green) - 1.0
    elif variant == "log":
        z_value = math.log(config.morph_k_log * p_green + 1.0)
    else:
        raise ValueError(f"Unsupported MorphMark variant: {config.morph_variant}")

    return min(max(z_value, config.morph_eps), 1.0 - config.morph_eps)


def _apply_morphmark(
    logits,
    green_mask,
    config: WatermarkConfig,
    eps: float = 1e-20,
) -> Tuple[object, Dict]:
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    probs = probs.clamp_min(eps)

    p_green_tensor = probs[green_mask].sum().clamp(min=eps, max=1.0 - eps)
    p_green = float(p_green_tensor.detach().cpu())
    r_value = _compute_morph_phi(config, p_green)

    green_probs = probs[green_mask]
    red_probs = probs[~green_mask]

    adjusted_probs = probs.clone()
    adjusted_probs[green_mask] = green_probs + (green_probs / p_green_tensor) * r_value * (1.0 - p_green_tensor)
    adjusted_probs[~green_mask] = red_probs * (1.0 - r_value)
    adjusted_probs = adjusted_probs.clamp_min(eps)
    adjusted_probs = adjusted_probs / adjusted_probs.sum().clamp_min(eps)
    adjusted_logits = torch.log(adjusted_probs)

    info = {
        "mode": "morph",
        "P_G": p_green,
        "r": r_value,
        "applied": r_value > config.morph_eps * 10,
    }
    return adjusted_logits, info


class WatermarkLogitsProcessor(LogitsProcessor):
    """Hugging Face-compatible logits processor for KGW / OPT / MorphMark."""

    def __init__(self, config: WatermarkConfig, vocab_size: int, hard: bool = False):
        self.config = config
        self.config.watermark_type = (self.config.watermark_type or "kgw").lower()
        if self.config.watermark_type not in SUPPORTED_WATERMARK_TYPES:
            raise ValueError(f"Unsupported watermark type: {self.config.watermark_type}")

        self.config.seeding_scheme = ensure_supported_seeding_scheme(self.config.seeding_scheme)
        self.vocab_size = vocab_size
        self._runtime_vocab_size = vocab_size
        self.hard = hard
        self.greenlist_cache = {}
        self.reset_metadata()

    def reset_metadata(self) -> None:
        self.step_metadata: List[Dict[str, object]] = []
        self.greenlist_sizes: List[int] = []
        self.spike_entropies: List[float] = []
        self.opt_B_values: List[float] = []
        self.opt_applied_flags: List[bool] = []
        self.morph_PG_values: List[float] = []
        self.morph_r_values: List[float] = []
        self.morph_applied_flags: List[bool] = []

    def _get_seed(self, prev_tokens: Sequence[int]) -> int:
        return _get_simple_seed(prev_tokens, self.config.hash_window, self.config.private_key or "")

    def _get_greenlist_ids_and_mask(self, seed: int, device):
        effective_vocab_size = int(getattr(self, "_runtime_vocab_size", self.vocab_size))
        cache_key = (seed, effective_vocab_size)

        if cache_key not in self.greenlist_cache:
            self.greenlist_cache[cache_key] = build_greenlist_ids(
                effective_vocab_size,
                seed,
                self.config.gamma,
                device="cpu",
            )

        green_ids = self.greenlist_cache[cache_key].to(device)
        green_mask = build_greenlist_mask(effective_vocab_size, green_ids, device=device)
        return green_ids, green_mask

    def _compute_kgw_spike_entropy(self, logits) -> float:
        z_modulus = (1 - self.config.gamma) * (math.exp(self.config.delta) - 1) / (
            1 + (math.exp(self.config.delta) - 1) * self.config.gamma
        )
        probs = logits.softmax(dim=-1)
        return compute_spike_entropy(probs, z_modulus)

    def _apply_row(
        self,
        logits,
        prev_tokens: Sequence[int],
        hard: bool = False,
        runtime_vocab_size: Optional[int] = None,
    ):
        import torch

        self._runtime_vocab_size = int(runtime_vocab_size or logits.shape[-1])
        seed = self._get_seed(prev_tokens)
        green_ids, green_mask = self._get_greenlist_ids_and_mask(seed, logits.device)
        del green_ids

        info: Dict[str, object] = {
            "watermark_type": self.config.watermark_type,
            "greenlist_size": int(green_mask.sum().item()),
            "applied": False,
            "spike_entropy": None,
        }

        if hard:
            masked_logits = logits.clone()
            masked_logits[~green_mask] = float("-inf")
            info.update({"mode": "hard", "applied": True})
            return masked_logits, info

        if self.config.watermark_type == "none":
            info["mode"] = "none"
            return logits, info

        if self.config.watermark_type == "kgw":
            info["spike_entropy"] = self._compute_kgw_spike_entropy(logits)
            biased_logits = _apply_kgw_bias(logits, green_mask, self.config.delta)
            info.update({"mode": "kgw", "applied": True})
            return biased_logits, info

        if self.config.watermark_type == "opt":
            b_value, gamma_t = _compute_opt_damage_B(logits, green_mask)
            info.update({"mode": "opt", "B": b_value, "Gamma_t": gamma_t})
            if b_value <= self.config.beta:
                masked_logits = logits.clone()
                masked_logits[~green_mask] = float("-inf")
                info["applied"] = True
                return masked_logits, info
            return logits, info

        if self.config.watermark_type == "morph":
            adjusted_logits, morph_info = _apply_morphmark(logits, green_mask, self.config)
            info.update(morph_info)
            return adjusted_logits, info

        raise ValueError(f"Unsupported watermark type: {self.config.watermark_type}")

    def _record_step_infos(self, batch_infos: Sequence[Dict[str, object]]) -> None:
        self.step_metadata.append({"batch": list(batch_infos)})
        for info in batch_infos:
            self.greenlist_sizes.append(int(info["greenlist_size"]))
            spike_entropy = info.get("spike_entropy")
            if spike_entropy is not None:
                self.spike_entropies.append(float(spike_entropy))

            if info.get("mode") == "opt":
                self.opt_B_values.append(float(info["B"]))
                self.opt_applied_flags.append(bool(info["applied"]))
            elif info.get("mode") == "morph":
                self.morph_PG_values.append(float(info["P_G"]))
                self.morph_r_values.append(float(info["r"]))
                self.morph_applied_flags.append(bool(info["applied"]))

    def __call__(self, input_ids, scores):
        runtime_vocab_size = int(scores.shape[-1])
        outputs, batch_infos = self._apply_batch(
            input_ids,
            scores,
            hard=self.hard,
            record_metadata=True,
            runtime_vocab_size=runtime_vocab_size,
        )
        del batch_infos
        return outputs

    def _apply_batch(
        self,
        input_ids,
        scores,
        hard: bool = False,
        record_metadata: bool = True,
        runtime_vocab_size: Optional[int] = None,
    ):
        import torch

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, vocab_size]")

        effective_vocab_size = int(runtime_vocab_size or scores.shape[-1])
        self._runtime_vocab_size = effective_vocab_size
        outputs = scores.clone()
        batch_infos: List[Dict[str, object]] = []
        for row_index in range(scores.size(0)):
            prev_tokens = input_ids[row_index].detach().cpu().tolist()
            updated_scores, info = self._apply_row(
                scores[row_index],
                prev_tokens,
                hard=hard,
                runtime_vocab_size=effective_vocab_size,
            )
            outputs[row_index] = updated_scores.to(outputs.dtype)
            batch_infos.append(info)

        if record_metadata:
            self._record_step_infos(batch_infos)

        return outputs, batch_infos

    def process_next_token_logits(
        self,
        logits,
        prev_tokens: Sequence[int],
        hard: bool = False,
        return_info: bool = False,
    ):
        import torch

        single_scores = logits.unsqueeze(0) if logits.ndim == 1 else logits
        input_ids = torch.as_tensor([list(prev_tokens)], dtype=torch.long, device=single_scores.device)
        processed_scores, batch_infos = self._apply_batch(
            input_ids,
            single_scores,
            hard=hard,
            record_metadata=False,
        )
        processed_scores = processed_scores.squeeze(0)
        info = batch_infos[0]
        return (processed_scores, info) if return_info else processed_scores

    def get_metadata(self) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "watermark_type": self.config.watermark_type,
            "num_steps": len(self.step_metadata),
            "greenlist_sizes": list(self.greenlist_sizes),
            "avg_greenlist_size": float(np.mean(self.greenlist_sizes)) if self.greenlist_sizes else None,
            "spike_entropies": list(self.spike_entropies),
            "avg_spike_entropy": float(np.mean(self.spike_entropies)) if self.spike_entropies else None,
            "opt_B_values": list(self.opt_B_values),
            "opt_applied_flags": list(self.opt_applied_flags),
            "morph_PG_values": list(self.morph_PG_values),
            "morph_r_values": list(self.morph_r_values),
            "morph_applied_flags": list(self.morph_applied_flags),
            "step_metadata": list(self.step_metadata),
        }

        if self.opt_B_values:
            metadata.update(
                {
                    "avg_opt_B": float(np.mean(self.opt_B_values)),
                    "opt_applied_tokens": int(sum(self.opt_applied_flags)),
                    "opt_applied_fraction": float(sum(self.opt_applied_flags) / len(self.opt_applied_flags)),
                }
            )

        if self.morph_PG_values:
            metadata.update(
                {
                    "avg_morph_P_G": float(np.mean(self.morph_PG_values)),
                    "avg_morph_r": float(np.mean(self.morph_r_values)),
                    "morph_applied_tokens": int(sum(self.morph_applied_flags)),
                    "morph_applied_fraction": float(sum(self.morph_applied_flags) / len(self.morph_applied_flags)),
                }
            )

        return metadata


class PrivateWatermarkLogitsProcessor(WatermarkLogitsProcessor):
    """Compatibility placeholder for the unsupported self-hash/private mode."""

    def __init__(self, config: WatermarkConfig, vocab_size: int, hard: bool = False):
        raise NotImplementedError(
            "private/selfhash mode is not implemented in this minimal baseline; "
            "use seeding_scheme='simple'."
        )


def hash_token(token, seed=0):
    """Hash function compatible with the original interface."""
    return hash_tokens([token], key=str(seed) if seed else "")


def watermark_sampling(logits, prev_token, gamma=0.5, delta=2.0, hard=False):
    """KGW watermark sampling helper compatible with the original interface."""
    config = WatermarkConfig(gamma=gamma, delta=delta, watermark_type="kgw")
    processor = WatermarkLogitsProcessor(config, logits.shape[-1], hard=hard)
    return processor.process_next_token_logits(logits, [prev_token], hard=hard)


def opt_watermark_sampling(logits, prev_token, gamma=0.5, beta=0.0):
    """OPT watermark sampling helper compatible with the original interface."""
    config = WatermarkConfig(gamma=gamma, beta=beta, watermark_type="opt")
    processor = WatermarkLogitsProcessor(config, logits.shape[-1])
    return processor.process_next_token_logits(logits, [prev_token], hard=False)


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
    """MorphMark helper compatible with the original interface."""
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
    return processor.process_next_token_logits(logits, [prev_token], hard=False)
