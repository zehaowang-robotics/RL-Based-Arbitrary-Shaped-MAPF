import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_DIR = Path("output") / "1-agents_CACTUS_PPO_QMIX_2026-04-13-19-52-42"
RESULT_FILE_RE = re.compile(r"results_(\d+)\.json$")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def result_epoch(path):
    match = RESULT_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_result_file(result_dir):
    candidates = [path for path in result_dir.glob("results_*.json") if result_epoch(path) is not None]
    if candidates:
        return max(candidates, key=result_epoch)
    final_result = result_dir / "results.json"
    if final_result.exists():
        return final_result
    raise FileNotFoundError(f"No results_*.json or results.json files found in {result_dir}")


def infer_epochs(data, result_file, default_interval):
    nr_points = len(data["success_rate"])
    latest_epoch = result_epoch(result_file)
    if latest_epoch is not None and nr_points > 1:
        interval = latest_epoch / float(nr_points - 1)
        return np.arange(nr_points, dtype=float) * interval
    return np.arange(nr_points, dtype=float) * float(default_interval)


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return np.arange(len(values)), values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.arange(window - 1, len(values)), np.convolve(values, kernel, mode="valid")


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def summarize(data, epochs, result_file):
    success = np.asarray(data["success_rate"], dtype=float)
    completion = np.asarray(data["completion_rate"], dtype=float)
    training_time = np.asarray(data.get("training_time", []), dtype=float)
    total_time = float(data.get("total_time", training_time.sum() if len(training_time) else 0.0))
    time_per_epoch = float(data.get("time_per_epoch", 0.0))
    window = min(10, len(success))

    best_success_index = int(success.argmax())
    best_completion_index = int(completion.argmax())
    summary = {
        "result_file": str(result_file),
        "points": int(len(success)),
        "final_epoch": float(epochs[-1]) if len(epochs) else 0.0,
        "total_training_hours": total_time / 3600.0,
        "time_per_epoch_seconds": time_per_epoch,
        "final_success_rate": float(success[-1]),
        "final_completion_rate": float(completion[-1]),
        "best_success_rate": float(success[best_success_index]),
        "best_success_epoch": float(epochs[best_success_index]),
        "best_completion_rate": float(completion[best_completion_index]),
        "best_completion_epoch": float(epochs[best_completion_index]),
        "first_window_success_mean": safe_mean(success[:window]),
        "last_window_success_mean": safe_mean(success[-window:]),
        "first_window_completion_mean": safe_mean(completion[:window]),
        "last_window_completion_mean": safe_mean(completion[-window:]),
        "last_window_success_std": float(success[-window:].std()) if window else float("nan"),
        "last_window_completion_std": float(completion[-window:].std()) if window else float("nan"),
    }
    summary["success_mean_delta"] = summary["last_window_success_mean"] - summary["first_window_success_mean"]
    summary["completion_mean_delta"] = summary["last_window_completion_mean"] - summary["first_window_completion_mean"]
    return summary


def save_points_csv(path, data, epochs):
    success = np.asarray(data["success_rate"], dtype=float)
    completion = np.asarray(data["completion_rate"], dtype=float)
    training_time = np.asarray(data.get("training_time", np.zeros_like(success)), dtype=float)
    auc_success = np.asarray(data.get("auc_success", np.zeros_like(success)), dtype=float)
    auc_completion = np.asarray(data.get("auc_completion", np.zeros_like(success)), dtype=float)
    cumulative_hours = np.cumsum(training_time) / 3600.0

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "success_rate",
                "completion_rate",
                "training_time_seconds",
                "cumulative_training_hours",
                "auc_success",
                "auc_completion",
            ]
        )
        for row in zip(epochs, success, completion, training_time, cumulative_hours, auc_success, auc_completion):
            writer.writerow([float(value) for value in row])


def plot_results(data, epochs, output_path, smoothing_window):
    success = np.asarray(data["success_rate"], dtype=float)
    completion = np.asarray(data["completion_rate"], dtype=float)
    training_time = np.asarray(data.get("training_time", np.zeros_like(success)), dtype=float)
    cumulative_hours = np.cumsum(training_time) / 3600.0
    cumulative_seconds = np.cumsum(training_time)
    auc_success = np.cumsum(np.asarray(data.get("auc_success", np.zeros_like(success)), dtype=float))
    auc_completion = np.cumsum(np.asarray(data.get("auc_completion", np.zeros_like(success)), dtype=float))
    weighted_success = np.divide(
        auc_success,
        cumulative_seconds,
        out=np.zeros_like(auc_success),
        where=cumulative_seconds > 0,
    )
    weighted_completion = np.divide(
        auc_completion,
        cumulative_seconds,
        out=np.zeros_like(auc_completion),
        where=cumulative_seconds > 0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Training Results")

    ax = axes[0, 0]
    ax.plot(epochs, success, "o-", color="#1f77b4", alpha=0.35, label="success raw")
    ax.plot(epochs, completion, "o-", color="#2ca02c", alpha=0.35, label="completion raw")
    smooth_index, smooth_success = moving_average(success, smoothing_window)
    _, smooth_completion = moving_average(completion, smoothing_window)
    smooth_window = min(smoothing_window, len(success))
    ax.plot(epochs[smooth_index], smooth_success, color="#1f77b4", linewidth=2.5, label=f"success MA{smooth_window}")
    ax.plot(epochs[smooth_index], smooth_completion, color="#2ca02c", linewidth=2.5, label=f"completion MA{smooth_window}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(cumulative_hours, success, "o-", color="#1f77b4", label="success")
    ax.plot(cumulative_hours, completion, "o-", color="#2ca02c", label="completion")
    ax.set_xlabel("cumulative training hours")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    width = max(1.0, (epochs[1] - epochs[0]) * 0.7 if len(epochs) > 1 else 1.0)
    ax.bar(epochs, training_time / 60.0, width=width, color="#9467bd", alpha=0.75)
    ax.set_xlabel("epoch")
    ax.set_ylabel("minutes per logged interval")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, weighted_success, color="#1f77b4", linewidth=2.0, label="time-weighted success")
    ax.plot(epochs, weighted_completion, color="#2ca02c", linewidth=2.0, label="time-weighted completion")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted average rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from CACTUS/PPO result JSON files.")
    parser.add_argument("result_dir", nargs="?", default=str(DEFAULT_RESULT_DIR), help="Directory containing results_*.json files.")
    parser.add_argument("--result-file", default=None, help="Specific result JSON file to plot. Defaults to the latest results_*.json.")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to <result_dir>/training_curves.png.")
    parser.add_argument("--csv", default=None, help="Output CSV path. Defaults to <result_dir>/training_points.csv.")
    parser.add_argument("--summary-json", default=None, help="Output summary JSON path. Defaults to <result_dir>/training_summary.json.")
    parser.add_argument("--default-epoch-interval", type=float, default=50.0, help="Epoch spacing used if it cannot be inferred from the filename.")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Moving-average window for plotted curves.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_file = Path(args.result_file) if args.result_file is not None else find_latest_result_file(result_dir)
    data = load_json(result_file)
    epochs = infer_epochs(data, result_file, args.default_epoch_interval)

    output_path = Path(args.out) if args.out is not None else result_dir / "training_curves.png"
    csv_path = Path(args.csv) if args.csv is not None else result_dir / "training_points.csv"
    summary_path = Path(args.summary_json) if args.summary_json is not None else result_dir / "training_summary.json"

    plot_results(data, epochs, output_path, args.smoothing_window)
    save_points_csv(csv_path, data, epochs)
    summary = summarize(data, epochs, result_file)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Loaded: {result_file}")
    print(f"Saved plot: {output_path}")
    print(f"Saved points: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Final success/completion: {summary['final_success_rate']:.3f}/{summary['final_completion_rate']:.3f}")
    print(f"Best success: {summary['best_success_rate']:.3f} at epoch {summary['best_success_epoch']:.0f}")
    print(f"Last-window success mean/std: {summary['last_window_success_mean']:.3f}/{summary['last_window_success_std']:.3f}")


if __name__ == "__main__":
    main()
