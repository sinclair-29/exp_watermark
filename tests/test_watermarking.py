from types import SimpleNamespace

import pytest

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import (
    WatermarkLogitsProcessor,
    _apply_kgw_bias,
)


def test_kgw_bias_only_green_tokens():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    green_mask = torch.tensor([True, False, True, False])

    updated_logits = _apply_kgw_bias(logits, green_mask, delta=2.0)

    assert torch.allclose(updated_logits[green_mask], logits[green_mask] + 2.0)
    assert torch.allclose(updated_logits[~green_mask], logits[~green_mask])


def test_opt_threshold_behavior(monkeypatch):
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    green_ids = torch.tensor([1, 3], dtype=torch.long)
    green_mask = torch.tensor([False, True, False, True])

    apply_config = WatermarkConfig(watermark_type="opt", gamma=0.5, beta=1e9)
    no_apply_config = WatermarkConfig(watermark_type="opt", gamma=0.5, beta=-1e9)

    apply_processor = WatermarkLogitsProcessor(apply_config, vocab_size=4)
    no_apply_processor = WatermarkLogitsProcessor(no_apply_config, vocab_size=4)

    monkeypatch.setattr(
        apply_processor,
        "_get_greenlist_ids_and_mask",
        lambda seed, device: (green_ids.to(device), green_mask.to(device)),
    )
    monkeypatch.setattr(
        no_apply_processor,
        "_get_greenlist_ids_and_mask",
        lambda seed, device: (green_ids.to(device), green_mask.to(device)),
    )

    applied_logits, applied_info = apply_processor.process_next_token_logits(
        logits,
        prev_tokens=[5],
        return_info=True,
    )
    unchanged_logits, unchanged_info = no_apply_processor.process_next_token_logits(
        logits,
        prev_tokens=[5],
        return_info=True,
    )

    assert torch.isinf(applied_logits[~green_mask]).all()
    assert torch.allclose(applied_logits[green_mask], logits[green_mask])
    assert applied_info["applied"] is True

    assert torch.allclose(unchanged_logits, logits)
    assert unchanged_info["applied"] is False
    assert not torch.isnan(applied_logits[green_mask]).any()
    assert not torch.isnan(unchanged_logits).any()


def test_morphmark_probability_mass(monkeypatch):
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    green_ids = torch.tensor([0, 2], dtype=torch.long)
    green_mask = torch.tensor([True, False, True, False])

    config = WatermarkConfig(watermark_type="morph", gamma=0.5, morph_variant="exp")
    processor = WatermarkLogitsProcessor(config, vocab_size=4)
    monkeypatch.setattr(
        processor,
        "_get_greenlist_ids_and_mask",
        lambda seed, device: (green_ids.to(device), green_mask.to(device)),
    )

    updated_logits, info = processor.process_next_token_logits(logits, prev_tokens=[1], return_info=True)
    original_probs = torch.softmax(logits, dim=-1)
    updated_probs = torch.softmax(updated_logits, dim=-1)

    old_p_green = original_probs[green_mask].sum().item()
    new_p_green = updated_probs[green_mask].sum().item()
    new_p_red = updated_probs[~green_mask].sum().item()
    expected_green = old_p_green + info["r"] * (1.0 - old_p_green)
    expected_red = (1.0 - old_p_green) * (1.0 - info["r"])

    assert torch.isclose(updated_probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert new_p_green == pytest.approx(expected_green, abs=1e-6)
    assert new_p_red == pytest.approx(expected_red, abs=1e-6)


@pytest.mark.parametrize("watermark_type", ["kgw", "opt", "morph"])
def test_processor_handles_logits_wider_than_tokenizer_vocab(watermark_type):
    torch = pytest.importorskip("torch")

    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    config = WatermarkConfig(
        watermark_type=watermark_type,
        gamma=0.5,
        delta=2.0,
        beta=0.0,
        morph_variant="exp",
    )

    # Simulate the old caveat: the processor is initialized from a tokenizer
    # vocab that is smaller than the actual logits dimension.
    processor = WatermarkLogitsProcessor(config, vocab_size=4)
    updated_scores = processor(input_ids, scores)

    assert updated_scores.shape == scores.shape
    assert processor.greenlist_sizes[-1] <= scores.shape[-1]
