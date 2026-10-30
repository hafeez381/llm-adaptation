"""One plot style for every report figure: 8-9 pt text at the final column width, no titles, thin marks.
Categorical colours follow the validated default palette in fixed slot order (dataviz skill, palette.md).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
WIDTH = 5.5  # inches, one NeurIPS text column

plt.rcParams.update({
    # body font of the NeurIPS template (Times) for all figure text, including math
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    "pdf.fonttype": 42})


def tag(ax, letter: str, label: str = "") -> None:
    ax.text(-0.14, 1.04, f"({letter}) {label}".strip(), transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def save(fig, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
