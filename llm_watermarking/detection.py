import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import (
    build_greenlist_ids,
    ensure_supported_seeding_scheme,
    hash_tokens,
)


def _flatten_token_ids(full_ids) -> List[int]:
    if hasattr(full_ids, "detach"):
        full_ids = full_ids.detach()
    if hasattr(full_ids, "cpu"):
        full_ids = full_ids.cpu()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()

    if isinstance(full_ids, list) and full_ids and isinstance(full_ids[0], list):
        if len(full_ids) != 1:
            raise ValueError("detect_tokens currently expects a single sequence or batch size 1.")
        full_ids = full_ids[0]

    return [int(token_id) for token_id in full_ids]


class WatermarkDetector:
    def __init__(
        self,
        tokenizer,
        config: Optional[WatermarkConfig] = None,
        watermark_vocab_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or WatermarkConfig()
        self.config.seeding_scheme = ensure_supported_seeding_scheme(self.config.seeding_scheme)
        self.watermark_vocab_size = int(watermark_vocab_size) if watermark_vocab_size is not None else None

    def _get_seed(self, prev_tokens: Sequence[int]) -> int:
        context_tokens = list(prev_tokens[-self.config.hash_window :]) if self.config.hash_window > 0 else []
        return hash_tokens(context_tokens, key=self.config.private_key or "")

    def _get_watermark_vocab_size(self) -> int:
        if self.watermark_vocab_size is not None:
            return self.watermark_vocab_size
        if self.tokenizer is None or not hasattr(self.tokenizer, "vocab_size"):
            raise ValueError("watermark_vocab_size is required when tokenizer.vocab_size is unavailable.")
        return int(self.tokenizer.vocab_size)

    def _get_greenlist_ids(self, prev_tokens: Sequence[int]):
        seed = self._get_seed(prev_tokens)
        return self._build_greenlist_ids(seed)

    def _build_greenlist_ids(self, seed: int):
        return build_greenlist_ids(
            self._get_watermark_vocab_size(),
            seed,
            self.config.gamma,
            device="cpu",
        )

    def _detect_from_tokens(
        self,
        tokens: Sequence[int],
        prompt_len: int = 0,
        ignore_repeated_ngrams: bool = True,
        return_details: bool = False,
        repeated_ngram_width: Optional[int] = None,
    ) -> Dict:
        prompt_len = max(0, min(int(prompt_len), len(tokens)))
        generated_len = max(0, len(tokens) - prompt_len)

        seen_ngrams = set()
        green_count = 0
        total_count = 0
        ignored_repeated_ngrams = 0
        token_results = []

        repeat_width = self.config.hash_window if repeated_ngram_width is None else max(0, int(repeated_ngram_width))

        for position in range(prompt_len, len(tokens)):
            if ignore_repeated_ngrams:
                start = max(0, position - repeat_width)
                ngram = tuple(tokens[start : position + 1])
                if ngram in seen_ngrams:
                    ignored_repeated_ngrams += 1
                    continue
                seen_ngrams.add(ngram)

            prev_tokens = tokens[:position]
            seed = self._get_seed(prev_tokens)
            watermark_vocab_size = self._get_watermark_vocab_size()
            green_ids = self._build_greenlist_ids(seed)
            green_lookup = set(green_ids.tolist())

            token_id = int(tokens[position])
            is_green = token_id in green_lookup
            if is_green:
                green_count += 1
            total_count += 1

            if return_details:
                token_results.append(
                    {
                        "position": position,
                        "token_index": position - prompt_len,
                        "absolute_token_position": position,
                        "token_id": token_id,
                        "token": self.tokenizer.decode([token_id]) if self.tokenizer is not None else None,
                        "seed": int(seed),
                        "watermark_vocab_size": watermark_vocab_size,
                        "is_green": is_green,
                        "cumulative_green_count": green_count,
                        "cumulative_scored_token_count": total_count,
                    }
                )

        if total_count == 0:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "prediction": False,
                "num_tokens": 0,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0.0,
                "expected_green_fraction": self.config.gamma,
                "prompt_len": prompt_len,
                "generated_len": generated_len,
                "watermark_vocab_size": self._get_watermark_vocab_size(),
                "ignored_repeated_ngrams": ignored_repeated_ngrams,
                "token_details": token_results if return_details else None,
            }

        expected = self.config.gamma * total_count
        variance = total_count * self.config.gamma * (1 - self.config.gamma)
        z_score = (green_count - expected) / np.sqrt(variance)
        p_value = 1 - norm.cdf(z_score)

        result = {
            "z_score": float(z_score),
            "p_value": float(p_value),
            "prediction": bool(p_value < 0.01),
            "num_tokens": total_count,
            "num_tokens_scored": total_count,
            "num_green_tokens": green_count,
            "green_fraction": green_count / total_count,
            "expected_green_fraction": self.config.gamma,
            "prompt_len": prompt_len,
            "generated_len": generated_len,
            "watermark_vocab_size": self._get_watermark_vocab_size(),
            "ignored_repeated_ngrams": ignored_repeated_ngrams,
        }

        if return_details:
            result["token_details"] = token_results

        return result

    def detect_tokens(
        self,
        full_ids,
        prompt_len: int = 0,
        ignore_repeated_ngrams: bool = True,
        return_details: bool = False,
    ) -> Dict:
        tokens = _flatten_token_ids(full_ids)
        return self._detect_from_tokens(
            tokens,
            prompt_len=prompt_len,
            ignore_repeated_ngrams=ignore_repeated_ngrams,
            return_details=return_details,
            repeated_ngram_width=self.config.hash_window,
        )

    def detect(
        self,
        text: str,
        ignore_repeated_ngrams: bool = True,
        ngram_size: int = 2,
        return_details: bool = False,
    ) -> Dict:
        """
        Legacy compatibility helper for whole-sequence scoring.

        This path tokenizes the entire input text and scores the resulting token
        sequence from the first token onward, so prompt tokens may be included in
        the score. For prompt+completion evaluation the preferred API is
        ``detect_tokens(full_ids, prompt_len=...)``.
        """
        warnings.warn(
            "WatermarkDetector.detect(text) scores the whole tokenized text. "
            "Use detect_tokens(full_ids, prompt_len=...) for prompt+completion evaluation.",
            UserWarning,
            stacklevel=2,
        )
        return self.detect_full_text_legacy(
            text,
            ignore_repeated_ngrams=ignore_repeated_ngrams,
            ngram_size=ngram_size,
            return_details=return_details,
        )

    def detect_full_text_legacy(
        self,
        text: str,
        ignore_repeated_ngrams: bool = True,
        ngram_size: int = 2,
        return_details: bool = False,
    ) -> Dict:
        """Tokenize and score the entire text. Prefer detect_tokens for experiments."""
        tokens = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        return self._detect_from_tokens(
            tokens,
            prompt_len=0,
            ignore_repeated_ngrams=ignore_repeated_ngrams,
            return_details=return_details,
            repeated_ngram_width=ngram_size,
        )

    def detect_with_multiple_keys(
        self,
        text: str,
        keys: List[str],
        significance_level: float = 0.01,
    ) -> Dict:
        results = []
        corrected_alpha = significance_level / len(keys)

        for key in keys:
            self.config.private_key = key
            result = self.detect(text)
            result["key"] = key
            results.append(result)

        best_result = min(results, key=lambda item: item["p_value"])

        return {
            "best_result": best_result,
            "all_results": results,
            "corrected_alpha": corrected_alpha,
            "watermark_detected": best_result["p_value"] < corrected_alpha,
            "bonferroni_correction_applied": True,
        }


def detect_watermark(text, tokenizer, gamma=0.5):
    """Detection helper compatible with the original interface."""
    config = WatermarkConfig(gamma=gamma)
    detector = WatermarkDetector(tokenizer, config)
    result = detector.detect(text)
    return result["z_score"], result["p_value"]
