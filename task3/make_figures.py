"""Task 3 figures and LaTeX tables, built only from saved result files.

python -m task3.make_figures   ->  report/figures/t3_*.pdf, report/tables/t3_*.tex
"""
import numpy as np

from common.plot import SERIES, WIDTH, plt, save, tag
from common.results import ROOT, read_json, read_jsonl

FIG, TAB = ROOT / "report/figures", ROOT / "report/tables"
NS = [1, 2, 4, 8, 16, 32]
NAMES = {"first": "FirstAns", "sc": "Self-consistency", "logp_high": "Highest log-prob", "sc_high": "Highest self-certainty",
         "sc_rwc": "Self-certainty + RWC", "shortest": "Shortest rollout", "vis_high": "Highest vision",
         "vis_rwc": "Vision + RWC", "visconf_high": "Highest VisConf", "visconf_rwc": "VisConf + RWC",
         "tokfuse_high": "Highest tokenwise fusion", "tokfuse_rwc": "Tokenwise fusion + RWC"}
DS = {"math": "MATH", "mathvista": "MathVista"}


def summ(ds):
    return read_json(ROOT / f"results/task3/selection/{ds}_summary.json")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def fig_selection():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    for ax, ds, letter in zip(axes, DS, "ab"):
        s = summ(ds)
        lines = [("oracle", "Oracle coverage", SERIES[7], "--"), ("sc", NAMES["sc"], SERIES[0], "-"),
                 ("sc_rwc", NAMES["sc_rwc"], SERIES[1], "-"), ("sc_high", NAMES["sc_high"], SERIES[2], "-"),
                 ("first", NAMES["first"], SERIES[3], ":")]
        if ds == "mathvista":
            lines.append(("visconf_rwc", NAMES["visconf_rwc"], SERIES[6], "-"))
        for m, label, c, ls in lines:
            key = "sc" if m == "oracle" else m
            ys = [s[f"heldout|{n}|{key}"]["oracle" if m == "oracle" else "acc"] for n in NS]
            ax.plot(NS, ys, ls, color=c, marker="o", label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.set_xlabel("Candidates $N$")
        tag(ax, letter, DS[ds])
    axes[0].set_ylabel("Held-out accuracy / coverage")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24))
    save(fig, FIG / "t3_selection.pdf")


def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    for ax, ds, letter in zip(axes, DS, "ab"):
        s = summ(ds)
        q = "sc_rwc" if ds == "math" else "visconf_rwc"
        for m, c in (("sc", SERIES[0]), (q, SERIES[1])):
            ax.plot([s[f"heldout|{n}|{m}"]["cost"] for n in NS], [s[f"heldout|{n}|{m}"]["acc"] for n in NS],
                    "-o", color=c, label="Self-consistency, fixed $N$" if m == "sc" else "Quality score + RWC, fixed $N$")
        a = ROOT / f"results/task3/adaptive/{ds}_heldout_summary.json"
        if a.exists():
            a = read_json(a)
            for name, c, mk in (("agreement", SERIES[2], "s"), ("quality", SERIES[6], "D")):
                ax.plot(a[name]["cost"], a[name]["acc"], mk, color=c, markersize=6, label=f"Adaptive, {name}")
        ax.set_xscale("log")
        ax.set_xlabel("Mean generated tokens per question")
        tag(ax, letter, DS[ds])
    axes[0].set_ylabel("Held-out accuracy")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    save(fig, FIG / "t3_cost.pdf")


def fig_traj():
    sig = read_json(ROOT / "results/task3/signals/mathvista.json")["trajectories"]
    fig, axes = plt.subplots(len(sig), 1, figsize=(WIDTH, 1.0 * len(sig) + 0.5), sharex=False)
    for ax, t, letter in zip(axes, sig, "abc"):
        c, v = np.array(t["c"]), np.array(t["v"])
        ax.plot(c, color=SERIES[0], lw=0.8, label="Calibrated confidence $c_t$")
        ax.plot(v, color=SERIES[1], lw=0.8, label="Visual engagement $v_t$")
        ax.set_ylim(0, 1)
        tag(ax, letter, f"MathVista question {t['sid']}, rollout {t['rollout']}, {'correct' if t['correct'] else 'incorrect'}")
    axes[-1].set_xlabel("Generated token")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, FIG / "t3_trajectories.pdf")


def table_methods():
    rows = []
    boot = {ds: read_json(ROOT / f"results/task3/selection/{ds}_bootstrap.json") for ds in DS}
    for m in NAMES:
        cells = []
        for ds in DS:
            s = summ(ds)
            if f"heldout|32|{m}" not in s:
                cells += ["--", "--", "--"]
                continue
            b = boot[ds].get(f"32|{m}-sc")
            ci = f"{100 * b['diff']:+.1f} [{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}]" if b else "--"
            cells += [pct(s[f"heldout|8|{m}"]["acc"]), pct(s[f"heldout|32|{m}"]["acc"]), ci]
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    s = {ds: summ(ds) for ds in DS}
    rows.append(r"\midrule Oracle coverage & " + " & ".join(
        f"{pct(s[ds]['heldout|8|sc']['oracle'])} & {pct(s[ds]['heldout|32|sc']['oracle'])} & --" for ds in DS) + r" \\")
    head = (r"\begin{tabular}{lcccccc}\toprule & \multicolumn{3}{c}{MATH} & \multicolumn{3}{c}{MathVista} \\"
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Selection rule & $N{=}8$ & $N{=}32$ & $\Delta$ vs SC at 32 & $N{=}8$ & $N{=}32$ & $\Delta$ vs SC at 32 \\ \midrule")
    write(TAB / "t3_methods.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_failures():
    rows = []
    for ds in DS:
        s = summ(ds)
        q = "sc_rwc" if ds == "math" else "visconf_rwc"
        for n in (1, 4, 32):
            a, b = s[f"heldout|{n}|sc"], s[f"heldout|{n}|{q}"]
            rows.append(f"{DS[ds]} & {n} & {pct(a['oracle'])} & {pct(a['fail_sampling'])} & {pct(a['acc'])} & "
                        f"{pct(a['fail_selection'])} & {pct(b['acc'])} & {pct(b['fail_selection'])} & {a['n_groups']:.2f}" + r" \\")
    head = (r"\begin{tabular}{lcccccccc}\toprule Dataset & $N$ & Coverage & Sampling fail & SC acc & SC selection fail & "
            r"RWC acc & RWC selection fail & Answer groups \\ \midrule")
    write(TAB / "t3_failures.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_signals():
    rows = []
    for ds in DS:
        sig = read_json(ROOT / f"results/task3/signals/{ds}.json")
        for k, label in (("L", "Mean log-prob $L$"), ("K", "Self-certainty $K$"), ("V", "Vision $V$"),
                         ("Z", "VisConf $Z$"), ("Ztok", "Tokenwise fusion $Z^{\\mathrm{token}}$")):
            if k in sig["spearman_length_vs_score"]:
                rows.append(f"{DS[ds]} & {label} & {sig['spearman_length_vs_score'][k]['mean']:+.2f} & "
                            f"{100 * sig['quartile_gap_correctness'][k]:+.1f}" + r" \\")
        rows.append(f"{DS[ds]} & Length $T$ & -- & {100 * sig['quartile_gap_correctness']['length']:+.1f}" + r" \\")
    head = (r"\begin{tabular}{llcc}\toprule Dataset & Signal & Spearman with length & Top minus bottom quartile "
            r"correctness (pp) \\ \midrule")
    write(TAB / "t3_signals.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}")


def table_adaptive():
    rows = []
    th = read_json(ROOT / "task3/configs/adaptive_thresholds.json")
    for ds in DS:
        p = ROOT / f"results/task3/adaptive/{ds}_heldout_summary.json"
        if not p.exists():
            continue
        a = read_json(p)
        s = summ(ds)
        for name in ("always32", "agreement", "quality"):
            r = a[name]
            sel = th[ds]["selected"].get(name, {})
            stops = r["stop_counts"]
            tot = sum(stops.values())
            dist = "/".join(pct(stops.get(str(k), stops.get(k, 0)) / tot, 0) for k in (4, 8, 16, 32))
            label = {"always32": "Fixed 32", "agreement": f"Agreement $\\tau{{=}}{sel.get('tau')}$",
                     "quality": f"Quality $\\tau{{=}}{sel.get('tau')}, \\gamma{{=}}{sel.get('gamma')}$"}[name]
            extra = (f"{r.get('incorrect_early_stops', 0)} / {r.get('incorrect_early_where_sc32_correct', 0)}"
                     if name != "always32" else "--")
            rows.append(f"{DS[ds]} & {label} & {pct(r['acc'])} & {r['cost']:.0f} & {r['rollouts']:.1f} & {dist} & {extra}" + r" \\")
        n8 = s["heldout|8|sc"]
        rows.append(f"{DS[ds]} & Fixed 8 (SC) & {pct(n8['acc'])} & {n8['cost']:.0f} & 8.0 & -- & --" + r" \\ \midrule")
    head = (r"\begin{tabular}{llccccc}\toprule Dataset & Policy & Accuracy & Tokens & Rollouts & Stops at 4/8/16/32 (\%) & "
            r"Wrong early / recoverable \\ \midrule")
    write(TAB / "t3_adaptive.tex", head + "\n" + "\n".join(rows).removesuffix(r" \midrule") + "\n" + r"\bottomrule\end{tabular}")


def table_live():
    p = ROOT / "results/task3/live/questions.jsonl"
    if not p.exists():
        return
    q = read_jsonl(p)
    calls = read_jsonl(ROOT / "results/task3/live/calls.jsonl")
    by = {}
    for c in calls:
        by.setdefault(c["call"], []).append(c)
    n = len(q)

    def stats(names):
        cs = [c for k in names for c in by[k]]
        return (sum(c["input_tokens"] for c in cs) / n, sum(c["output_tokens"] for c in cs) / n,
                sum(c["seconds"] for c in cs) / n, np.mean([c["truncated"] for c in cs]),
                np.mean([c["answer"] is None for c in cs]))
    strat = {"Single (3,072)": (["single"], "single_correct"),
             "Parallel 3 x 1,024": (["parallel_1", "parallel_2", "parallel_3"], "parallel_correct"),
             "Sequential 1,024 + 2,048": (["parallel_1", "revision"], "revision_correct")}
    rows = []
    for name, (names, key) in strat.items():
        inp, out, sec, tr, ne = stats(names)
        rows.append(f"{name} & {pct(np.mean([r[key] for r in q]))} & {inp:.0f} & {out:.0f} & {sec:.1f} & {pct(tr)} & {pct(ne)}" + r" \\")
    w2c = sum(not r["initial_correct"] and r["revision_correct"] for r in q)
    c2w = sum(r["initial_correct"] and not r["revision_correct"] for r in q)
    head = (r"\begin{tabular}{lcccccc}\toprule Strategy & Accuracy & Input tokens & Output tokens & Seconds & "
            r"Truncated (\%) & No answer (\%) \\ \midrule")
    write(TAB / "t3_live.tex", head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" +
          f"\n% transitions vs initial answer: wrong->correct {w2c}, correct->wrong {c2w}, initial accuracy "
          f"{pct(np.mean([r['initial_correct'] for r in q]))}")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    fig_selection()
    fig_cost()
    fig_traj()
    table_methods()
    table_failures()
    table_signals()
    table_adaptive()
    table_live()
    print("written", sorted(p.name for p in FIG.glob("t3_*")), sorted(p.name for p in TAB.glob("t3_*")))


if __name__ == "__main__":
    main()
