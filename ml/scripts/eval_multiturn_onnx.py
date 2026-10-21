"""Compare int8 ONNX graphs on chat eval v2 as plain messages and as multi-turn windows.

Usage (from ml/):
    uv run python scripts/eval_multiturn_onnx.py \\
        distilbert_default:models/distilbert:256:0.30 \\
        distilbert_student_s42:models/students/minilm_l6_s42:128:0.25
"""

# Parse CLI options.
import json
import sys
from pathlib import Path

from secure_chat_ml.baseline import load_chat_style_eval_set
from secure_chat_ml.distilbert import load_saved_classifier
from secure_chat_ml.multiturn import build_multiturn_frame
from secure_chat_ml.onnx_eval import metrics_at_threshold, score_onnx


def main() -> None:
    """Each argument is export_name:tokenizer_dir:max_length:threshold."""

    frame = load_chat_style_eval_set(Path("data/chat_eval/chat_style_eval_v2.csv"))
    pool = frame.loc[frame["label"] == 0, "text"].astype(str).tolist()
    # Average over a few window seeds so one random draw does not decide the result.
    windows = [build_multiturn_frame(frame, pool, seed=seed) for seed in (1, 2, 3)]
    labels = frame["label"].to_numpy().astype(int)
    report: dict[str, object] = {}
    for spec in sys.argv[1:]:
        name, tok_dir, max_length, threshold = spec.split(":")
        _model, tokenizer = load_saved_classifier(Path(tok_dir))
        onnx = Path("exports/onnx_web") / name / "model.onnx"
        plain = metrics_at_threshold(
            labels,
            score_onnx(onnx, tokenizer, frame["text"].tolist(), max_length=int(max_length)),
            float(threshold),
        )
        accs = []
        for window in windows:
            scores = score_onnx(
                onnx, tokenizer, window["text"].tolist(), max_length=int(max_length)
            )
            accs.append(metrics_at_threshold(labels, scores, float(threshold))["accuracy"])
        report[name] = {
            "plain_accuracy": plain["accuracy"],
            "multiturn_accuracy_mean": sum(accs) / 3,
        }
        print(name, report[name])
    Path("reports/compression").mkdir(parents=True, exist_ok=True)
    Path("reports/compression/multiturn_onnx.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
