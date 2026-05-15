import pytest

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.detection import WatermarkDetector


class SimpleTokenizer:
    def __init__(self, vocab_size=16, eos_token_id=15):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def __call__(self, text, return_tensors="pt"):
        torch = pytest.importorskip("torch")
        ids = [int(token) for token in text.split()] if text.strip() else [0]
        return type("Batch", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token != self.eos_token_id]
        return " ".join(str(int(token)) for token in token_ids)


def test_detection_excludes_prompt(monkeypatch):
    torch = pytest.importorskip("torch")
    tokenizer = SimpleTokenizer(vocab_size=10, eos_token_id=9)
    detector = WatermarkDetector(tokenizer, WatermarkConfig(gamma=0.5, hash_window=1))

    monkeypatch.setattr(
        detector,
        "_get_greenlist_ids",
        lambda prev_tokens: torch.tensor([6], dtype=torch.long),
    )

    prompt_ids = [1, 2]
    generated_ids = [6, 6, 6]
    full_ids = prompt_ids + generated_ids

    result = detector.detect_tokens(full_ids, prompt_len=len(prompt_ids), ignore_repeated_ngrams=False)

    assert result["prompt_len"] == len(prompt_ids)
    assert result["generated_len"] == len(generated_ids)
    assert result["num_tokens_scored"] == len(generated_ids)
    assert result["num_green_tokens"] == len(generated_ids)
