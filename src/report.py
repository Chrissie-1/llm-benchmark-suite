"""Turn results/benchmark.csv into the chart and the README table."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TASK_LABEL = {"mmlu": "MMLU (5-shot)", "gsm8k": "GSM8K (8-shot)"}
MODEL_ORDER = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Phi-3-mini"]
MMLU_CHANCE = 0.25

# Slots 1-3 of the reference categorical palette, validated all-pairs in both
# modes. Light-mode aqua sits under 3:1 on the surface, so every mark carries a
# visible direct label (the relief rule).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#8a8880",
        "grid": "#e3e2dd",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#8a8880",
        "grid": "#343430",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def style_axes(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=theme["secondary"], length=0, labelsize=9)
    ax.yaxis.grid(True, color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)


def accuracy_panel(ax, df, theme):
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    tasks = [t for t in ("mmlu", "gsm8k") if t in set(df["task"])]
    width = 0.34
    positions = range(len(models))

    for i, task in enumerate(tasks):
        offset = (i - (len(tasks) - 1) / 2) * width
        values = [
            df[(df.model == m) & (df.task == task)]["accuracy"].mean() for m in models
        ]
        xs = [p + offset for p in positions]
        # 2px surface-coloured edge keeps a gap between adjacent bars.
        ax.bar(
            xs, values, width * 0.92, label=TASK_LABEL[task],
            color=theme["series"][i], edgecolor=theme["surface"], linewidth=2, zorder=3,
        )
        for x, value in zip(xs, values):
            ax.text(
                x, value + 0.02, f"{value:.2f}", ha="center", va="bottom",
                fontsize=9, color=theme["primary"], zorder=4,
            )

    # Reserve empty margin on the right so the chance annotation never sits on a bar.
    ax.set_xlim(-0.62, len(models) - 1 + 0.95)
    if "mmlu" in tasks:
        ax.axhline(MMLU_CHANCE, color=theme["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax.text(
            len(models) - 1 + 0.9, MMLU_CHANCE, "chance\n(4-way)",
            fontsize=8, color=theme["muted"], ha="right", va="center", linespacing=1.3,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", color=theme["secondary"], fontsize=9)
    ax.set_title("Accuracy by task", color=theme["primary"], fontsize=11, loc="left", pad=10)
    legend = ax.legend(
        frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])


def tradeoff_panel(ax, df, theme):
    gen = df[df.task == "gsm8k"]
    models = [m for m in MODEL_ORDER if m in set(gen["model"])]
    for i, model in enumerate(models):
        row = gen[gen.model == model].iloc[0]
        ax.plot(
            row["tok_per_s"], row["accuracy"], marker="o", markersize=11,
            color=theme["series"][i], markeredgecolor=theme["surface"],
            markeredgewidth=2, linestyle="none", zorder=3,
        )
        # Label to the right: throughput barely varies, so the points stack
        # vertically and side labels cannot collide.
        ax.annotate(
            model, (row["tok_per_s"], row["accuracy"]),
            textcoords="offset points", xytext=(14, 0), ha="left", va="center",
            fontsize=9, color=theme["primary"], zorder=4,
        )
    # From zero: these throughputs differ by a fraction of a token per second, and
    # a zoomed axis would inflate that into an apparent speed ranking.
    ax.set_xlim(0, max(gen["tok_per_s"]) * 1.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Generation throughput (tokens/s)", color=theme["secondary"], fontsize=9)
    ax.set_ylabel("GSM8K accuracy", color=theme["secondary"], fontsize=9)
    ax.set_title(
        "Accuracy vs. speed", color=theme["primary"], fontsize=11, loc="left", pad=10
    )


def render(df, mode, out_path):
    theme = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=theme["surface"])
    for ax in axes:
        style_axes(ax, theme)
    accuracy_panel(axes[0], df, theme)
    tradeoff_panel(axes[1], df, theme)
    quant = df["quant"].iloc[0]
    fig.suptitle(
        f"Small-model evaluation - {quant.upper()} on RTX 4060 (8 GB), batch size 1",
        color=theme["secondary"], fontsize=9.5, x=0.008, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def summary_table(df):
    """One row per model: both accuracies, generation speed, peak memory."""
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(df["model"])]:
        sub = df[df.model == model]
        gen = sub[sub.task == "gsm8k"]
        mmlu = sub[sub.task == "mmlu"]
        rows.append({
            "Model": model,
            "MMLU (5-shot)": f"{mmlu['accuracy'].iloc[0]:.2f}" if len(mmlu) else "-",
            "GSM8K (8-shot)": f"{gen['accuracy'].iloc[0]:.2f}" if len(gen) else "-",
            "Gen tok/s": f"{gen['tok_per_s'].iloc[0]:.1f}" if len(gen) else "-",
            "Peak VRAM": f"{sub['peak_vram_gb'].max():.2f} GB",
        })
    return pd.DataFrame(rows)


PENDING_NOTE = """
> **Phi-3-mini is pending.** The model loads and runs correctly; it is waiting on
> host memory, not on a fix. Loading it needs ~7.6 GB of commit charge to mmap its
> safetensors shards, and this machine has a fixed pagefile — see
> [Notes on this hardware](#notes-on-this-hardware). `src/retry_phi3.ps1` waits for
> headroom and merges its rows into the CSV, after which the chart and this table
> are regenerated. The rows above are final and unaffected.
"""

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def observation(df):
    """One line on the spread, written from whatever models are actually present."""
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    if len(models) < 2:
        return ""
    first, last = models[0], models[-1]
    parts = []
    for task, label in (("mmlu", "MMLU"), ("gsm8k", "GSM8K")):
        lo = df[(df.model == first) & (df.task == task)]["accuracy"]
        hi = df[(df.model == last) & (df.task == task)]["accuracy"]
        if len(lo) and len(hi):
            parts.append(f"{hi.iloc[0] - lo.iloc[0]:+.2f} {label}")
    if not parts:
        return ""
    return f"\nGoing from {first} to {last} buys {' and '.join(parts)}.\n"


def update_readme(df, readme=ROOT / "README.md"):
    """Rewrite the marked results block so the table never drifts from the CSV."""
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README markers missing - skipped")
        return

    block = [summary_table(df).to_markdown(index=False)]
    if "Phi-3-mini" not in set(df["model"]):
        block.append(PENDING_NOTE)
    block.append(observation(df))

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    # newline="\n": without it Windows translates to CRLF, so regenerating an
    # unchanged report rewrites every line of the file.
    readme.write_text(
        f"{head}{START}\n" + "\n".join(b for b in block if b) + f"\n{END}{tail}",
        encoding="utf-8",
        newline="\n",
    )
    print(f"updated {readme}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(RESULTS / "benchmark.csv"))
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    render(df, "light", RESULTS / "benchmark.png")
    render(df, "dark", RESULTS / "benchmark-dark.png")

    table = summary_table(df)
    (RESULTS / "summary.md").write_text(
        table.to_markdown(index=False) + "\n", encoding="utf-8", newline="\n"
    )
    if not args.no_readme:
        update_readme(df)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
