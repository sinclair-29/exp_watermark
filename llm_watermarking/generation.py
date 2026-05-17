from typing import Any, Dict, Optional

from llm_watermarking.config import WatermarkConfig
from llm_watermarking.watermarking import WatermarkLogitsProcessor

try:
    from transformers import LogitsProcessorList
except ImportError:  # pragma: no cover - exercised only in envs without transformers.
    class LogitsProcessorList(list):  # type: ignore[override]
        """Fallback list implementation so the package can import without transformers."""

        def __call__(self, input_ids, scores):
            for processor in self:
                scores = processor(input_ids, scores)
            return scores


def _get_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            return None
    return None


def _get_batch_value(batch, key: str, default=None):
    if isinstance(batch, dict):
        return batch.get(key, default)
    return getattr(batch, key, default)


def _maybe_to_device(tensor, device):
    if tensor is None or device is None or not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def _tensor_to_list(tokens):
    if hasattr(tokens, "detach"):
        tokens = tokens.detach()
    if hasattr(tokens, "cpu"):
        tokens = tokens.cpu()
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    return list(tokens)


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_tokenizer_len(tokenizer):
    try:
        return int(len(tokenizer))
    except (TypeError, AttributeError):
        return None


class WatermarkBeamSearcher:
    """
    Thin compatibility wrapper over the main HF generation path.

    The repository now uses ``model.generate(...)`` for the real implementation.
    This class remains for compatibility with older imports and simply runs the
    same generation path with beam search settings.
    """

    def __init__(self, model, tokenizer, config: WatermarkConfig, num_beams: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.num_beams = num_beams

    def generate(self, prompt: str, max_new_tokens: int = 50, hard: bool = False) -> str:
        result = generate_with_watermark(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            config=self.config,
            hard=hard,
            do_sample=False,
            num_beams=self.num_beams,
        )
        return result["full_text"]


def generate_with_watermark(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    config: Optional[WatermarkConfig] = None,
    hard: bool = False,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """
    Generate a completion and return prompt/completion tokenization explicitly.

    The returned ids are decoded without re-tokenizing text so detection and PPL
    can operate on the exact generated token sequence.
    """
    if config is None:
        config = WatermarkConfig()

    device = _get_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = _maybe_to_device(_get_batch_value(encoded, "input_ids"), device)
    attention_mask = _maybe_to_device(_get_batch_value(encoded, "attention_mask"), device)

    prompt_ids = _tensor_to_list(input_ids[0])
    prompt_len = len(prompt_ids)

    model_config = getattr(model, "config", None)
    model_config_vocab_size = _safe_int(getattr(model_config, "vocab_size", None))
    model_vocab_size = model_config_vocab_size
    if model_vocab_size is None:
        model_vocab_size = _safe_int(getattr(model, "vocab_size", None))
    tokenizer_vocab_size = _safe_int(getattr(tokenizer, "vocab_size", None))
    tokenizer_len = _safe_tokenizer_len(tokenizer)
    fallback_vocab_size = model_vocab_size or tokenizer_vocab_size or tokenizer_len

    logits_processor = None
    watermark_processor = None
    if config.watermark_type.lower() != "none":
        if fallback_vocab_size is None:
            raise ValueError("Unable to infer a vocabulary size for watermarking.")
        watermark_processor = WatermarkLogitsProcessor(config, int(fallback_vocab_size), hard=hard)
        logits_processor = LogitsProcessorList([watermark_processor])

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
        if getattr(tokenizer, "pad_token_id", None) is None:
            generate_kwargs["pad_token_id"] = eos_token_id

    output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
    full_ids = _tensor_to_list(output_ids[0])
    generated_ids = full_ids[prompt_len:]
    generated_len = len(generated_ids)

    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metadata: Dict[str, Any] = {
        "watermark_type": config.watermark_type,
        "model_config_vocab_size": model_config_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_len": tokenizer_len,
        "watermark_vocab_size": fallback_vocab_size,
        "generation_method": "hf_generate",
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "hard": hard,
        "full_ids": full_ids,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "prompt_len": prompt_len,
        "generated_len": generated_len,
    }

    if watermark_processor is not None:
        metadata.update(watermark_processor.get_metadata())

    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "full_ids": full_ids,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "prompt_len": prompt_len,
        "generated_len": generated_len,
        "metadata": metadata,
    }
