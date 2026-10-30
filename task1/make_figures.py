"""Task 1 figures and LaTeX tables, built only from saved result files.

python -m task1.make_figures   ->  report/figures/t1_*.pdf, report/tables/t1_*.tex
"""
import numpy as np

from common.plot import SERIES, WIDTH, plt, save, tag
from common.results import ROOT, read_json, read_jsonl
from task1.merge import CONDITIONS

FIG, TAB = ROOT / "report/figures", ROOT / "report/tables"
RUNS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
LABEL = {"baseline": "Baseline", "full_ft@625": "Full FT (update 625)", "full_ft": "Full FT", "l2sp": "Full FT + L2-SP",
         "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
BENCH = [("gsm8k_cot", "exact_match,flexible-extract", None, "GSM8K"), ("hellaswag", "acc_norm,none", "acc,none", "HellaSwag"),
         ("boolq", "acc,none", None, "BoolQ"), ("arc_easy", "acc_norm,none", "acc,none", "ARC-Easy")]


def run_dir(m):
    return ROOT / "results/task1" / m


def smooth(y, w=25):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode="valid")


def fig_loss():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    for m, c in zip(RUNS, SERIES):
        rows = read_jsonl(run_dir(m) / "train_log.jsonl")
        y = smooth([r["sft_loss"] for r in rows])
        axes[0].plot(np.arange(len(y)) + 25, y, color=c, lw=1.2, ls="--" if m == "l2sp" else "-", label=LABEL[m])
    axes[0].set_xlabel("Optimizer update")
    axes[0].set_ylabel("SFT loss (25-update mean)")
    axes[0].legend()
    tag(axes[0], "a")
    rows = read_jsonl(run_dir("l2sp") / "train_log.jsonl")
    u = [r["update"] for r in rows]
    axes[1].plot(u, [r["lam_omega"] for r in rows], color=SERIES[1], label=r"$\lambda\Omega$ (L2-SP penalty)")
    axes[1].plot(u, smooth([r["sft_loss"] for r in rows], 1), color=SERIES[0], lw=0.6, alpha=0.6, label="SFT loss (L2-SP run)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Optimizer update")
    axes[1].legend()
    tag(axes[1], "b")
    save(fig, FIG / "t1_loss.pdf")


def table_bench():
    """Main table: primary scores with changes from the baseline. Appendix table: primary and raw accuracy."""
    agg = {c: read_json(ROOT / f"results/task1/eval/{c}/aggregates.json") for c in CONDITIONS}
    base = agg["baseline"]
    main, full = [], []
    for c in CONDITIONS:
        cells, raws = [], []
        for task, prim, raw, _ in BENCH:
            v = agg[c][task][prim]
            d = "" if c == "baseline" else f" ({100 * (v - base[task][prim]):+.1f})".replace("-", "$-$")
            cells.append(f"{100 * v:.1f}{d}")
            raws.append(f"{100 * v:.1f} & {100 * agg[c][task][raw]:.1f}" if raw else f"{100 * v:.1f}")
        main.append(f"{LABEL[c]} & " + " & ".join(cells) + f" & {100 * agg[c]['gsm8k_truncation_rate']:.1f}" + r" \\")
        full.append(f"{LABEL[c]} & " + " & ".join(raws) + r" \\")
    head = (r"\begin{tabular}{lccccc}\toprule Condition & GSM8K & HellaSwag & BoolQ & ARC-Easy & "
            r"GSM8K truncated (\%) \\ \midrule")
    write(TAB / "t1_bench.tex", head + "\n" + "\n".join(main) + "\n" + r"\bottomrule\end{tabular}")
    head = (r"\begin{tabular}{lcccccc}\toprule & GSM8K & \multicolumn{2}{c}{HellaSwag} & BoolQ & \multicolumn{2}{c}{ARC-Easy} \\ "
            r"Condition & Exact match & Normalized & Raw & Accuracy & Normalized & Raw \\ \midrule")
    write(TAB / "t1_bench_full.tex", head + "\n" + "\n".join(full) + "\n" + r"\bottomrule\end{tabular}")


def table_resources():
    """Method-specific overhead is stated in the appendix text, so the table stays at natural width."""
    rows = []
    disp = read_json(ROOT / "results/task1/displacement.json")
    for m in RUNS:
        r = read_json(run_dir(m) / "resources.json")
        cs = r["checkpoint_sizes"]
        d = f"{disp[m]:.1f}" if m in disp else "--"
        rows.append(f"{LABEL[m]} & {r['trainable_params'] / 1e6:.2f} M & {r['train_seconds'] / 3600:.2f} & "
                    f"{r['peak_memory_bytes'] / 1e9:.1f} & {cs['final_bytes'] / 1e9:.3f} & {cs['resume_bytes'] / 1e9:.2f} & {d}" + r" \\")
    head = (r"\begin{tabular}{lcccccc}\toprule Run & Trainable & Hours & Peak memory (GB) & Final checkpoint (GB) & "
            r"Resume checkpoint (GB) & $\lVert\theta-\theta_0\rVert_2$ \\ \midrule")
    write(TAB / "t1_resources.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_diag():
    """One row per model; each cell lists layers 0 / 13 / 27, so the table stays narrow."""
    d = read_json(ROOT / "results/task1/diagnostics/weight_activation.json")
    rows = []
    for m in ["full_ft@625", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]:
        rs = [next(r for r in d["rows"] if r["model"] == m and r["layer"] == l) for l in d["layers"]]
        cell = lambda key, fmt: " / ".join(format(r[key], fmt) for r in rs)
        rows.append(f"{LABEL[m]} & {cell('rho_k', '.3f')} & {cell('eta', '.3f')} & "
                    f"{' / '.join(format(100 * r['R_math_mean'], '.1f') for r in rs)} & "
                    f"{' / '.join(format(100 * r['R_nonmath_mean'], '.1f') for r in rs)}" + r" \\")
    head = (r"\begin{tabular}{lcccc}\toprule Model & $\rho_k$ & $\eta$ & $R$ math (\%) & $R$ non-math (\%) \\ \midrule")
    write(TAB / "t1_diagnostics.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_boolq():
    b = read_json(ROOT / "results/task1/boolq_bias.json")
    rows = [f"{LABEL[c]} & {100 * b[c]['pred_yes']:.1f} & {100 * b[c]['acc_gold_yes']:.1f} & {100 * b[c]['acc_gold_no']:.1f}"
            + r" \\" for c in CONDITIONS]
    head = r"\begin{tabular}{lccc}\toprule Condition & Predicted yes (\%) & Acc. gold yes & Acc. gold no \\ \midrule"
    write(TAB / "t1_boolq.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_qual():
    s = read_json(ROOT / "results/task1/qual/annotation_summary.json")["per_condition"]
    cols = ["format_violation", "incoherent", "repetition", "math_framing", "inappropriate", "truncated"]
    rows = [f"{LABEL[c]} & " + " & ".join(str(s[c][k]) for k in cols) + r" \\" for c in CONDITIONS]
    head = (r"\begin{tabular}{lcccccc}\toprule Condition & Format & Incoherent & Repetition & Math framing & "
            r"Inappropriate & Truncated \\ \midrule")
    write(TAB / "t1_qual.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    for f in (fig_loss, table_bench, table_resources, table_diag, table_boolq, table_qual):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
    print(sorted(p.name for p in FIG.glob("t1_*")), sorted(p.name for p in TAB.glob("t1_*")))


if __name__ == "__main__":
    main()
