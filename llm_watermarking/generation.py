from typing import Dict, Optional, Tuple

import numpy as np

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import (
    PrivateWatermarkLogitsProcessor,
    WatermarkLogitsProcessor,
    compute_spike_entropy,
)


class WatermarkBeamSearcher:
    def __init__(self, model, tokenizer, config: WatermarkConfig, num_beams: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.num_beams = num_beams
        self.watermark_processor = WatermarkLogitsProcessor(config, tokenizer.vocab_size)

    def generate(self, prompt: str, max_new_tokens: int = 50, hard: bool = False) -> str:
        """
        Generate watermarked text using Beam Search.
        """
        import torch

        device = self.model.device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        beams = [(0.0, input_ids[0].tolist())]

        for _ in range(max_new_tokens):
            all_candidates = []

            for score, tokens in beams:
                input_tensor = torch.tensor([tokens], device=device)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits[:, -1, :].squeeze(0)

                logits = self.watermark_processor(logits, tokens, hard=hard)
                log_probs = torch.log_softmax(logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, self.num_beams)

                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_tokens = tokens + [token_id.item()]
                    new_score = score + log_prob.item()
                    all_candidates.append((new_score, new_tokens))

            all_candidates.sort(key=lambda item: item[0], reverse=True)
            beams = all_candidates[: self.num_beams]

            if all(self.tokenizer.eos_token_id in tokens for _, tokens in beams):
                break

        best_tokens = beams[0][1]
        return self.tokenizer.decode(best_tokens, skip_special_tokens=True)


def generate_with_watermark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    config: Optional[WatermarkConfig] = None,
    hard: bool = False,
    use_beam_search: bool = False,
    num_beams: int = 4,
) -> Tuple[str, Dict]:
    import torch

    if config is None:
        config = WatermarkConfig()

    if use_beam_search:
        searcher = WatermarkBeamSearcher(model, tokenizer, config, num_beams)
        text = searcher.generate(prompt, max_new_tokens, hard)
        return text, {"method": "beam_search", "num_beams": num_beams}

    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    if config.seeding_scheme == "private":
        processor = PrivateWatermarkLogitsProcessor(config, tokenizer.vocab_size)
    else:
        processor = WatermarkLogitsProcessor(config, tokenizer.vocab_size)

    spike_entropies = []
    opt_B_values = []
    opt_applied_count = 0

    morph_PG_values = []
    morph_r_values = []
    morph_applied_count = 0

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :].squeeze(0)

        prev_tokens = generated[0].tolist()

        probs = torch.softmax(logits, dim=-1)
        z_mod = (1 - config.gamma) * (np.exp(config.delta) - 1) / (
            1 + (np.exp(config.delta) - 1) * config.gamma
        )
        spike_entropy = compute_spike_entropy(probs, z_mod)
        spike_entropies.append(spike_entropy)

        if config.seeding_scheme == "private":
            logits, forced_token = processor(logits, prev_tokens, hard)
            if forced_token is not None:
                next_token = torch.tensor([[forced_token]], device=device)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        else:
            logits, wm_info = processor(logits, prev_tokens, hard, return_info=True)
            mode = wm_info.get("mode")
            if mode == "opt":
                opt_B_values.append(wm_info.get("B"))
                opt_applied_count += int(wm_info.get("applied", False))
            elif mode == "morph":
                morph_PG_values.append(wm_info.get("P_G"))
                morph_r_values.append(wm_info.get("r"))
                morph_applied_count += int(wm_info.get("applied", False))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)

    metadata = {
        "method": "multinomial",
        "watermark_type": config.watermark_type,
        "avg_spike_entropy": np.mean(spike_entropies) if spike_entropies else 0,
        "num_tokens_generated": len(spike_entropies),
    }
    if config.watermark_type.lower() == "opt":
        metadata.update(
            {
                "beta": config.beta,
                "avg_opt_B": float(np.mean(opt_B_values)) if opt_B_values else None,
                "opt_applied_tokens": opt_applied_count,
                "opt_applied_fraction": opt_applied_count / max(1, len(opt_B_values)),
            }
        )
    elif config.watermark_type.lower() == "morph":
        metadata.update(
            {
                "morph_variant": config.morph_variant,
                "morph_p0": config.morph_p0,
                "avg_morph_P_G": float(np.mean(morph_PG_values)) if morph_PG_values else None,
                "avg_morph_r": float(np.mean(morph_r_values)) if morph_r_values else None,
                "morph_applied_tokens": morph_applied_count,
                "morph_applied_fraction": morph_applied_count / max(1, len(morph_r_values)),
            }
        )

    return text, metadata
