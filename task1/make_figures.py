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
        axes[0].plot(np.arange(len(y)) + 25, y, color=c, lw=1.2, label=LABEL[m])
    axes[0].set_xlabel("Optimizer update")
    axes[0].set_ylabel("SFT loss (25-update mean)")
    axes[0].legend(fontsize=6.5)
    tag(axes[0], "a")
    rows = read_jsonl(run_dir("l2sp") / "train_log.jsonl")
    u = [r["update"] for r in rows]
    axes[1].plot(u, [r["lam_omega"] for r in rows], color=SERIES[1], label=r"$\lambda\Omega$ (L2-SP penalty)")
    axes[1].plot(u, smooth([r["sft_loss"] for r in rows], 1), color=SERIES[0], lw=0.6, alpha=0.6, label="SFT loss (L2-SP run)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Optimizer update")
    axes[1].legend(fontsize=6.5)
    tag(axes[1], "b")
    save(fig, FIG / "t1_loss.pdf")


def table_bench():
    agg = {c: read_json(ROOT / f"results/task1/eval/{c}/aggregates.json") for c in CONDITIONS}
    base = agg["baseline"]
    rows = []
    for c in CONDITIONS:
        cells = []
        for task, prim, raw, _ in BENCH:
            v = agg[c][task][prim]
            d = "" if c == "baseline" else f" ({100 * (v - base[task][prim]):+.1f})"
            extra = f" / {100 * agg[c][task][raw]:.1f}" if raw else ""
            cells.append(f"{100 * v:.1f}{d}{extra}")
        cells.append(f"{100 * agg[c]['gsm8k_truncation_rate']:.1f}")
        rows.append(f"{LABEL[c]} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{lccccc}\toprule Condition & GSM8K flex. EM & HellaSwag acc\_norm / acc & BoolQ acc & "
            r"ARC-Easy acc\_norm / acc & GSM8K truncated (\%) \\ \midrule")
    write(TAB / "t1_bench.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_resources():
    rows = []
    disp = read_json(ROOT / "results/task1/displacement.json") if (ROOT / "results/task1/displacement.json").exists() else {}
    for m in RUNS:
        r = read_json(run_dir(m) / "resources.json")
        cs = r["checkpoint_sizes"]
        final = cs.get("final_bytes", 0) / 1e9
        oh = r["overhead"]
        note = (f"$\\theta_0$ {oh['theta0_bytes'] / 1e9:.1f} GB" if "theta0_bytes" in oh else
                f"SVD {oh['svd_seconds']:.0f} s, bases {oh['basis_bytes'] / 1e6:.0f} MB" if "svd_seconds" in oh else "--")
        d = f"{disp[m]:.1f}" if m in disp else "--"
        rows.append(f"{LABEL[m]} & {r['trainable_params'] / 1e6:.2f} M & {r['train_seconds'] / 3600:.2f} & "
                    f"{r['peak_memory_bytes'] / 1e9:.1f} & {final:.3f} & {cs.get('resume_bytes', 0) / 1e9:.2f} & {d} & {note}" + r" \\")
    head = (r"\begin{tabular}{lccccccl}\toprule Run & Trainable & Train hours & Peak GB & Final ckpt GB & Resume ckpt GB & "
            r"$\lVert\theta-\theta_0\rVert_2$ & Method overhead \\ \midrule")
    write(TAB / "t1_resources.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_diag():
    d = read_json(ROOT / "results/task1/diagnostics/weight_activation.json")
    rows = []
    for m in ["full_ft@625", "full_ft", "l2sp", "lora4", "lora16", "oplora16"]:
        rs = {r["layer"]: r for r in d["rows"] if r["model"] == m}
        cells = []
        for l in d["layers"]:
            r = rs[l]
            cells.append(f"{r['rho_k']:.3f} & {r['eta']:.4f} & {r['R_math_mean']:.4f} & {r['R_nonmath_mean']:.4f}")
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    hdr = " & ".join(rf"\multicolumn{{4}}{{c}}{{Layer {l}}}" for l in d["layers"])
    sub = " & ".join([r"$\rho_k$ & $\eta$ & $R$ math & $R$ other"] * len(d["layers"]))
    head = (r"\begin{tabular}{l" + "cccc" * len(d["layers"]) + r"}\toprule & " + hdr + r" \\ Model & " + sub + r" \\ \midrule")
    write(TAB / "t1_diagnostics.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    for f in (fig_loss, table_bench, table_resources, table_diag):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)
    print(sorted(p.name for p in FIG.glob("t1_*")), sorted(p.name for p in TAB.glob("t1_*")))


if __name__ == "__main__":
    main()
