from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import hash_tokens, partition_vocab


class WatermarkDetector:
    def __init__(self, tokenizer, config: Optional[WatermarkConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or WatermarkConfig()

    def _get_seed(self, prev_tokens: List[int]) -> int:
        """Get seed according to the configuration."""
        window = prev_tokens[-self.config.hash_window :]
        return hash_tokens(window, key=self.config.private_key or "")

    def detect(
        self,
        text: str,
        ignore_repeated_ngrams: bool = True,
        ngram_size: int = 2,
        return_details: bool = False,
    ) -> Dict:
        """
        Detect watermark in text.

        Args:
            text: text to be detected
            ignore_repeated_ngrams: whether to ignore repeated n-grams (Paper Section 4.1)
            ngram_size: n-gram size
            return_details: whether to return detailed information

        Returns:
            Detection result dict
        """
        tokens = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

        if len(tokens) <= self.config.hash_window:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "prediction": False,
                "num_tokens": 0,
                "green_fraction": 0.0,
            }

        seen_ngrams = set()
        green_count = 0
        total_count = 0
        token_results = []

        for i in range(self.config.hash_window, len(tokens)):
            if ignore_repeated_ngrams:
                ngram = tuple(tokens[i - ngram_size : i + 1]) if i >= ngram_size else tuple(tokens[: i + 1])
                if ngram in seen_ngrams:
                    continue
                seen_ngrams.add(ngram)

            prev_tokens = tokens[:i]
            seed = self._get_seed(prev_tokens)
            green, _ = partition_vocab(self.tokenizer.vocab_size, seed, self.config.gamma)

            is_green = tokens[i] in green
            if is_green:
                green_count += 1
            total_count += 1

            if return_details:
                token_results.append(
                    {
                        "position": i,
                        "token_id": tokens[i],
                        "token": self.tokenizer.decode([tokens[i]]),
                        "is_green": is_green,
                    }
                )

        if total_count == 0:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "prediction": False,
                "num_tokens": 0,
                "green_fraction": 0.0,
            }

        expected = self.config.gamma * total_count
        variance = total_count * self.config.gamma * (1 - self.config.gamma)
        z_score = (green_count - expected) / np.sqrt(variance)
        p_value = 1 - norm.cdf(z_score)

        result = {
            "z_score": z_score,
            "p_value": p_value,
            "prediction": p_value < 0.01,
            "num_tokens": total_count,
            "num_green_tokens": green_count,
            "green_fraction": green_count / total_count,
            "expected_green_fraction": self.config.gamma,
        }

        if return_details:
            result["token_details"] = token_results

        return result

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
    """Detection function compatible with the original interface."""
    config = WatermarkConfig(gamma=gamma)
    detector = WatermarkDetector(tokenizer, config)
    result = detector.detect(text)
    return result["z_score"], result["p_value"]
