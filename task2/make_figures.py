"""Task 2 figures, tables and the D15 example selection, built only from saved result files.

python -m task2.make_figures   ->  report/figures/t2_*.pdf, report/tables/t2_*.tex, results/task2/examples.json
D15 rule: per suite and outcome class under the correct and complementary conditions - (a) correct on both images,
(b) the same answer for both images, (c) the swap follows the partner image's answer - the pair with the smallest
pair id.
"""
import numpy as np

from common.plot import SERIES, WIDTH, plt, save, tag
from common.results import ROOT, read_json, read_jsonl, write_json
from task2.final_eval import CONDITIONS
from task2.vqa_norm import normalize

FIG, TAB = ROOT / "report/figures", ROOT / "report/tables"
RES = ROOT / "results/task2"
BOUNDS = [2250, 3750]
SUITES = {"grounding": "Annotation-derived", "vqa": "Natural VQAv2"}


def fig_curves():
    tr = read_jsonl(RES / "train/train_log.jsonl")
    dev = read_jsonl(RES / "train/dev_metrics.jsonl")
    u = [r["update"] for r in dev]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6))
    loss = np.convolve([r["loss"] for r in tr], np.ones(25) / 25, mode="valid")
    axes[0].plot(np.arange(len(loss)) + 25, loss, color=SERIES[0], lw=1, label="Training loss")
    axes[0].plot(u, [r["caption_loss"] for r in dev], "o-", color=SERIES[1], label="Dev caption loss")
    axes[0].set_ylabel("Loss")
    for i, (suite, key) in enumerate((("Annotation-derived", "grounding_dev"), ("Natural VQAv2", "vqa_dev"))):
        ls = "-" if i == 0 else "--"
        axes[1].plot(u, [r[key]["nll_correct"] for r in dev], ls, color=SERIES[0], marker="o", label=f"{suite}, correct")
        axes[1].plot(u, [r[key]["nll_no_image"] for r in dev], ls, color=SERIES[3], marker="o", label=f"{suite}, no image")
        axes[2].plot(u, [r[key]["delta_nll"] for r in dev], ls, color=SERIES[2 if i == 0 else 6], marker="o", label=suite)
    axes[1].set_ylabel("Answer NLL")
    axes[2].set_ylabel(r"Image-swap penalty $\Delta_{\mathrm{NLL}}$")
    axes[2].axhline(0, color="#6b6a63", lw=0.6)
    for ax, letter in zip(axes, "abc"):
        for b in BOUNDS:
            ax.axvline(b, color="#6b6a63", lw=0.6, ls=":")
        ax.set_xlabel("Update")
        ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.36))
        tag(ax, letter)
    fig.tight_layout()
    save(fig, FIG / "t2_curves.pdf")


def table_final():
    s = read_json(RES / "final/summary.json")
    a = read_json(RES / "final/advantages.json")
    rows = []
    for suite, name in SUITES.items():
        for c in CONDITIONS:
            d = s[f"{suite}|{c}"]
            types = " / ".join(f"{100 * v:.1f}" for _, v in sorted(d["by_type"].items()))
            rows.append(f"{name} & {c.replace('_', ' ')} & {100 * d['accuracy']:.1f} & {types} & {100 * d['pair_accuracy']:.1f} & "
                        f"{100 * d['identical_answer_rate']:.1f} & {100 * d['empty_rate']:.1f} & {d['mean_reference_nll']:.2f} & "
                        f"{1000 * d['seconds_per_answer']:.0f}" + r" \\")
        acc, nll = a[f"{suite}|accuracy"], a[f"{suite}|nll"]
        comp = s[f"{suite}|complementary"]
        rows.append(rf"\multicolumn{{9}}{{l}}{{\footnotesize {name}: correct-image advantage {100 * acc['diff']:+.1f} pp "
                    rf"[{100 * acc['lo']:+.1f}, {100 * acc['hi']:+.1f}], NLL advantage {nll['diff']:+.2f} "
                    rf"[{nll['lo']:+.2f}, {nll['hi']:+.2f}]; under the swap {100 * comp['follows_partner']:.0f}\% follow the partner, "
                    rf"{100 * comp['neither']:.0f}\% match neither}} \\ \midrule")
    head = (r"\begin{tabular}{llccccccc}\toprule Suite & Image & Accuracy & By type & Pair acc. & Identical & Empty & "
            r"Ref. NLL & ms/answer \\ \midrule")
    write(TAB / "t2_final.tex", head + "\n" + "\n".join(rows).removesuffix(r" \midrule") + "\n" + r"\bottomrule\end{tabular}")


def table_main():
    """Compact main-text table: accuracy, pair accuracy and reference NLL per suite and image condition."""
    s = read_json(RES / "final/summary.json")
    rows = []
    for c in CONDITIONS:
        cells = []
        for suite in SUITES:
            d = s[f"{suite}|{c}"]
            cells.append(f"{100 * d['accuracy']:.1f} & {100 * d['pair_accuracy']:.1f} & {d['mean_reference_nll']:.2f}")
        rows.append(f"{c.replace('_', ' ').capitalize()} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lcccccc}\toprule & \multicolumn{3}{c}{Annotation-derived (64 pairs)} & "
            r"\multicolumn{3}{c}{Natural VQAv2 (96 pairs)} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} Image & Accuracy & "
            r"Pair acc. & Ref. NLL & Accuracy & Pair acc. & Ref. NLL \\ \midrule")
    write(TAB / "t2_main.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_resources():
    r = read_json(RES / "train/resources.json")
    fc = read_json(RES / "feature_cache.json")
    rows = [f"Feature caching & -- & {fc['encode_seconds'] / 60:.1f} & {(fc['peak_memory_bytes'] or 0) / 1e9:.2f} & "
            f"{fc['images']} images, {fc['storage_bytes'] / 1e9:.2f} GB FP16" + r" \\"]
    for k, name in (("0", "Stage 1 caption"), ("1", "Stage 2 instruction"), ("2", "Stage 3 VQA")):
        st = r["stages"][k]
        rows.append(f"{name} & {st['trainable_params'] / 1e6:.2f} M & {st['seconds'] / 60:.1f} & "
                    f"{st.get('peak_memory_bytes', 0) / 1e9:.2f} & --" + r" \\")
    head = r"\begin{tabular}{lcccl}\toprule Phase & Trainable & Minutes & Peak GB & Note \\ \midrule"
    write(TAB / "t2_resources.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def select_examples():
    rows = read_jsonl(RES / "final/rows.jsonl")
    out = {}
    for suite in SUITES:
        pairs = {}
        for r in rows:
            if r["suite"] == suite and r["condition"] in ("correct", "complementary"):
                pairs.setdefault(r["pair_id"], {}).setdefault(r["condition"], {})[r["slot"]] = r
        classes = {"correct_on_both": lambda p: all(p["correct"][k]["match_own"] for k in (0, 1)),
                   "same_answer_both_images": lambda p: normalize(p["correct"][0]["pred"]) == normalize(p["correct"][1]["pred"]),
                   "swap_follows_partner": lambda p: all(p["complementary"][k]["match_partner"] for k in (0, 1))}
        for cls, f in classes.items():
            hit = sorted(pid for pid, p in pairs.items() if f(p))
            if hit:
                p = pairs[hit[0]]
                out[f"{suite}|{cls}"] = {"pair_id": hit[0], "question": p["correct"][0]["question"], "count": len(hit),
                                         "images": [p["correct"][k]["image_id"] for k in (0, 1)],
                                         "references": [p["correct"][k]["reference"] for k in (0, 1)],
                                         "pred_correct": [p["correct"][k]["pred"] for k in (0, 1)],
                                         "pred_swap": [p["complementary"][k]["pred"] for k in (0, 1)]}
    write_json(RES / "examples.json", out)
    return out


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    for f in (fig_curves, table_final, table_main, table_resources, select_examples):
        f()
    print(sorted(p.name for p in FIG.glob("t2_*")), sorted(p.name for p in TAB.glob("t2_*")))


if __name__ == "__main__":
    main()
