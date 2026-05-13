# LLM Watermarking

This repository is a research artifact for KGW, OPT, and MorphMark watermarking experiments on causal language models.

The codebase is intentionally small and paper-friendly:
- `config.py` for experiment parameters
- `watermarking.py` for watermark construction and processors
- `generation.py` for decoding
- `detection.py` for detection
- `analysis.py` for bounds and attack simulation
- `cli.py` for the experiment entrypoint

## Environment

- Python 3.10+
- PyTorch
- Transformers
- NumPy
- SciPy

## Install

```bash
pip install -r requirements.txt
```

## Main Experiment

Run the paper workflow from the CLI:

```bash
python -m llm_watermarking --model_path ./models/Phi-3-mini-128k-instruct --prompt "The future of AI is"
```

See all flags with:

```bash
python -m llm_watermarking --help
```

## Programmatic Use

Use the main experiment API from the package root:

```python
from llm_watermarking import WatermarkConfig, generate_with_watermark
```

Lower-level helpers are available from their flat modules, for example:

```python
from llm_watermarking.watermarking import hash_tokens, watermark_sampling
from llm_watermarking.detection import detect_watermark
```
