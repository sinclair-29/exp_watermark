from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WatermarkConfig:
    """Watermark configuration parameters.

    Notes:
    - ``seeding_scheme="simple"`` means the green/red list for the next token is
      seeded from the previous ``hash_window`` tokens.
    - Detection must use the same seeding settings as generation or the recovered
      green/red lists will not match.
    - ``hash_window=1`` is the closest simple baseline to paper-style KGW / OPT /
      MorphMark experiments in this repository.
    """

    watermark_type: str = "kgw"
    gamma: float = 0.5
    delta: float = 2.0
    beta: float = 0.0
    hash_window: int = 1
    seeding_scheme: str = "simple"
    private_key: Optional[str] = None
    multiple_keys: Optional[List[str]] = None

    morph_variant: str = "exp"
    morph_p0: float = 0.15
    morph_eps: float = 1e-10
    morph_k_linear: float = 1.55
    morph_k_exp: float = 1.30
    morph_k_log: float = 2.15
