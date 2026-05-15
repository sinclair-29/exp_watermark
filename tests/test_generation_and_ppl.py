import math
from types import SimpleNamespace

import pytest

from llm_watermarking.analysis import compute_completion_logppl_and_ppl
from llm_watermarking.config import WatermarkConfig
from llm_watermarking.generation import generate_with_watermark


class FakeTokenizer:
    def __init__(self, vocab_size=8, eos_token_id=7):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.eos_token = str(eos_token_id)

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        torch = pytest.importorskip("torch")
        del add_special_tokens
        ids = [int(token) for token in text.split()] if text.strip() else [0]
        attention_mask = [1] * len(ids)
        return SimpleNamespace(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.tensor([attention_mask], dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token != self.eos_token_id]
        return " ".join(str(int(token)) for token in token_ids)


class FakeGenerateModel:
    def __init__(self, vocab_size=8):
        torch = pytest.importorskip("torch")
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.last_labels = None

    def _base_scores(self, input_ids):
        torch = pytest.importorskip("torch")
        last_token = int(input_ids[0, -1].item())
        return torch.tensor(
            [
                0.1 + 0.05 * ((last_token + 0) % 3),
                0.2 + 0.05 * ((last_token + 1) % 3),
                0.3 + 0.05 * ((last_token + 2) % 3),
                0.4,
                0.5,
                0.6,
                0.7,
                -1.0,
            ],
            dtype=torch.float32,
        ).view(1, -1)

    def generate(self, input_ids, max_new_tokens=5, logits_processor=None, eos_token_id=None, **kwargs):
        torch = pytest.importorskip("torch")
        del kwargs
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            scores = self._base_scores(generated)
            if logits_processor is not None:
                scores = logits_processor(generated, scores)
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
        return generated

    def __call__(self, input_ids, attention_mask=None, labels=None):
        torch = pytest.importorskip("torch")
        del attention_mask
        self.last_labels = labels.clone() if labels is not None else None
        return SimpleNamespace(loss=torch.tensor(0.5, dtype=torch.float32))


def test_generation_metadata_shapes():
    pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    model = FakeGenerateModel()

    result = generate_with_watermark(
        model,
        tokenizer,
        prompt="0 1",
        max_new_tokens=3,
        config=WatermarkConfig(watermark_type="kgw", gamma=0.5, delta=2.0),
        do_sample=False,
        num_beams=1,
    )

    assert len(result["generated_ids"]) == result["generated_len"]
    assert len(result["full_ids"]) == result["prompt_len"] + result["generated_len"]
    assert result["generated_text"] == tokenizer.decode(result["generated_ids"], skip_special_tokens=True)


def test_generation_uses_model_vocab_size_when_tokenizer_vocab_is_smaller():
    pytest.importorskip("torch")
    tokenizer = FakeTokenizer(vocab_size=4)
    model = FakeGenerateModel(vocab_size=8)

    result = generate_with_watermark(
        model,
        tokenizer,
        prompt="0 1",
        max_new_tokens=2,
        config=WatermarkConfig(watermark_type="kgw", gamma=0.5, delta=2.0),
        do_sample=False,
        num_beams=1,
    )

    assert len(result["generated_ids"]) == result["generated_len"]
    assert len(result["full_ids"]) == result["prompt_len"] + result["generated_len"]


def test_ppl_masks_prompt():
    torch = pytest.importorskip("torch")
    model = FakeGenerateModel()
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    result = compute_completion_logppl_and_ppl(model, input_ids, prompt_len=2)

    assert torch.equal(model.last_labels[:, :2], torch.full((1, 2), -100, dtype=torch.long))
    assert torch.equal(model.last_labels[:, 2:], input_ids[:, 2:])
    assert result["num_scored_tokens"] == 3
    assert result["ppl"] == pytest.approx(math.exp(0.5))
