"""Aggregate per-seed student reports into mean and std (reports/compression/)."""

# Group summaries by student family (directory name minus the _s<seed> suffix).
import json
import re
import statistics
from pathlib import Path


def main() -> None:
    """Print and save mean/std of TEST and chat-v1 metrics per student family."""

    groups: dict[str, list[dict]] = {}
    for path in sorted(Path("reports/students").glob("*/summary.json")):
        family = re.sub(r"_s\d+$", "", path.parent.name)
        groups.setdefault(family, []).append(json.loads(path.read_text()))
    out: dict[str, object] = {}
    for family, rows in groups.items():
        entry: dict[str, object] = {"seeds": [row["seed"] for row in rows]}
        for key in ("test_metrics", "chat_style_eval_metrics"):
            for metric in ("accuracy", "scam_f1"):
                values = [row[key][metric] for row in rows]
                entry[f"{key}.{metric}"] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
        entry["thresholds"] = [row["threshold"] for row in rows]
        out[family] = entry
    Path("reports/compression").mkdir(parents=True, exist_ok=True)
    Path("reports/compression/students_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
