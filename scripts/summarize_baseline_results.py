#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


METHOD_NAMES = {
    "none": "No Watermark",
    "kgw": "KGW",
    "opt": "OPT",
    "morph_linear": "MorphMark",
    "morph_exp": "MorphMark",
    "morph_log": "MorphMark",
}

VARIANT_NAMES = {
    "none": "-",
    "kgw": "-",
    "opt": "-",
    "morph_linear": "linear",
    "morph_exp": "exp",
    "morph_log": "log",
}

DISPLAY_ORDER = [
    ("Phi-3-mini", "none"),
    ("Phi-3-mini", "kgw"),
    ("Phi-3-mini", "opt"),
    ("Phi-3-mini", "morph_linear"),
    ("Phi-3-mini", "morph_exp"),
    ("Phi-3-mini", "morph_log"),
    ("Qwen2.5-7B", "none"),
    ("Qwen2.5-7B", "kgw"),
    ("Qwen2.5-7B", "opt"),
    ("Qwen2.5-7B", "morph_linear"),
    ("Qwen2.5-7B", "morph_exp"),
    ("Qwen2.5-7B", "morph_log"),
    ("Llama-2-7B", "none"),
    ("Llama-2-7B", "kgw"),
    ("Llama-2-7B", "opt"),
    ("Llama-2-7B", "morph_linear"),
    ("Llama-2-7B", "morph_exp"),
    ("Llama-2-7B", "morph_log"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize baseline watermark experiment results.")
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw JSON outputs.")
    parser.add_argument("--summary_csv", required=True, help="Path to write the CSV summary.")
    parser.add_argument("--summary_md", required=True, help="Path to write the Markdown summary.")
    return parser


def mean_or_nan(values: List[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def std_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return statistics.pstdev(values)


def scan_raw_results(raw_dir: Path) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {"z_scores": [], "ppls": []})

    for path in sorted(raw_dir.glob("*/*/*.json")):
        model_name = path.parts[-3]
        method_key = path.parts[-2]

        with path.open() as handle:
            payload = json.load(handle)

        z_score = payload["detection"]["z_score"]
        ppl_payload = payload.get("ppl") or {}
        ppl = ppl_payload.get("ppl")

        grouped[(model_name, method_key)]["z_scores"].append(float(z_score))
        if ppl is not None and math.isfinite(float(ppl)):
            grouped[(model_name, method_key)]["ppls"].append(float(ppl))

    return grouped


def build_rows(grouped: Dict[Tuple[str, str], Dict[str, List[float]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, method_key in DISPLAY_ORDER:
        stats = grouped.get((model_name, method_key), {"z_scores": [], "ppls": []})
        z_scores = stats["z_scores"]
        ppls = stats["ppls"]

        rows.append(
            {
                "Model": model_name,
                "Method": METHOD_NAMES[method_key],
                "Variant": VARIANT_NAMES[method_key],
                "Avg. z-score": mean_or_nan(z_scores),
                "Std. z-score": std_or_nan(z_scores),
                "Avg. PPL": mean_or_nan(ppls),
                "Std. PPL": std_or_nan(ppls),
                "Num Runs": len(z_scores),
            }
        )
    return rows


def format_float(value: float, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def write_csv(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Model",
                "Method",
                "Variant",
                "Avg. z-score",
                "Std. z-score",
                "Avg. PPL",
                "Std. PPL",
                "Num Runs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "Method": row["Method"],
                    "Variant": row["Variant"],
                    "Avg. z-score": format_float(float(row["Avg. z-score"]), 6),
                    "Std. z-score": format_float(float(row["Std. z-score"]), 6),
                    "Avg. PPL": format_float(float(row["Avg. PPL"]), 6),
                    "Std. PPL": format_float(float(row["Std. PPL"]), 6),
                    "Num Runs": row["Num Runs"],
                }
            )


def write_markdown(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("| Model | Method | Variant | Avg. z-score | Std. z-score | Avg. PPL | Std. PPL |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['Model']} | {row['Method']} | {row['Variant']} | "
                f"{format_float(float(row['Avg. z-score']), 4)} | "
                f"{format_float(float(row['Std. z-score']), 4)} | "
                f"{format_float(float(row['Avg. PPL']), 4)} | "
                f"{format_float(float(row['Std. PPL']), 4)} |\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    summary_csv = Path(args.summary_csv)
    summary_md = Path(args.summary_md)

    grouped = scan_raw_results(raw_dir)
    rows = build_rows(grouped)

    write_csv(rows, summary_csv)
    write_markdown(rows, summary_md)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
