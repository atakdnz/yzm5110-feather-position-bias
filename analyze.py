"""Builds every table and figure in the report from the saved outputs in results/.

python analyze.py  ->  results/summary.json, results/tables.md, figures/*.png
"""
import json, os, itertools
from collections import defaultdict
import numpy as np

RES, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
BUDGETS = [288, 144, 58, 29]
R_OF = {288: "50%", 144: "75%", 58: "90%", 29: "95%"}
MAIN = ["fastv", "cc4", "cc2", "grid", "rand", "fastv_avgq"]
LABEL = {"full": "Full (576)", "fastv": "FastV (last-token attn)", "fastv_avgq": "FastV (paper text: mean received attn)",
         "cc4": "CC-FastV 4x4 (proposed)", "cc2": "CC-FastV 2x2 (ablation)", "grid": "Uniform grid (no attn)",
         "rand": "Random (mean of 3 seeds)"}


def load(task):
    p = f"{RES}/preds_{task}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def ok_matrix(recs, name):
    if name.startswith("rand@"):
        k = name.split("@")[1]
        seeds = [s for s in range(3) if f"rand{s}@{k}" in recs[0]]
        return np.mean([[r[f"rand{s}@{k}"]["ok"] for s in seeds] for r in recs], axis=1)
    return np.array([r[name]["ok"] for r in recs], dtype=float)


def boot_ci(x, n=2000):
    idx = np.random.default_rng(0).integers(0, len(x), (n, len(x)))  # same resamples for every statistic
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar_p(a, b):
    """Exact two-sided McNemar on paired binary outcomes."""
    from scipy.stats import binomtest
    a, b = a.astype(bool), b.astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n10, n01 + n10, 0.5).pvalue), n01, n10


def names_for(m, k):
    return f"rand@{k}" if m == "rand" else f"{m}@{k}"


summary, md = {}, []

# ---------------- A-OKVQA ----------------
aok = load("aokvqa")
# extra control run (second VM): RoPE-free FastV score; fastv/cc4 re-run there as a determinism check
NR = f"{RES}/norope/preds_aokvqa.jsonl"
if aok and os.path.exists(NR):
    nr = {r["idx"]: r for r in map(json.loads, open(NR))}
    agree = {}
    for k in BUDGETS:
        for m in ("fastv", "cc4"):
            agree[f"{m}@{k}"] = float(np.mean([nr[r["idx"]][f"{m}@{k}"]["pred"] == r[f"{m}@{k}"]["pred"] for r in aok]))
        for r in aok:
            r[f"fastvnr@{k}"] = nr[r["idx"]][f"fastvnr@{k}"]
    summary["rerun_pred_agreement"] = agree
    MAIN.append("fastvnr")
    LABEL["fastvnr"] = "FastV, RoPE-free score (FEATHER-style)"
if aok:
    n = len(aok)
    S = {"n": n, "acc": {}, "acc_greedy": {}, "ci": {}, "vs_fastv": {}}
    full = ok_matrix(aok, "full")
    S["acc"]["full"] = 100 * full.mean()
    S["ci"]["full"] = [100 * v for v in boot_ci(full)]
    S["acc_greedy"]["full"] = 100 * np.mean([r["full"]["ok_greedy"] for r in aok])
    for k in BUDGETS:
        f = ok_matrix(aok, f"fastv@{k}")
        for m in MAIN:
            x = ok_matrix(aok, names_for(m, k))
            S["acc"][f"{m}@{k}"] = 100 * x.mean()
            S["ci"][f"{m}@{k}"] = [100 * v for v in boot_ci(x)]
            if m != "rand":
                S["acc_greedy"][f"{m}@{k}"] = 100 * np.mean([r[f"{m}@{k}"]["ok_greedy"] for r in aok])
            if m != "fastv":
                d = x - f
                lo, hi = boot_ci(d)
                entry = dict(diff=100 * d.mean(), ci=[100 * lo, 100 * hi])
                if m != "rand":
                    p, n01, n10 = mcnemar_p(f, x)
                    entry.update(p_mcnemar=p, fastv_wrong_m_right=n01, fastv_right_m_wrong=n10)
                S["vs_fastv"][f"{m}@{k}"] = entry
        rs = [100 * np.mean([r[f"rand{s}@{k}"]["ok"] for r in aok]) for s in range(3)]
        S["acc"][f"rand_seeds@{k}"] = rs
        # proposed rule against the attention-free coverage controls (paired)
        c4 = ok_matrix(aok, f"cc4@{k}")
        for m in ("grid", "rand"):
            d = c4 - ok_matrix(aok, names_for(m, k))
            lo, hi = boot_ci(d)
            S.setdefault("cc4_vs", {})[f"{m}@{k}"] = dict(diff=100 * d.mean(), ci=[100 * lo, 100 * hi])
        # how many questions the full model gets right are lost by each rule
        fm = full.astype(bool)
        S.setdefault("lost_of_full_correct", {})
        for m in ("fastv", "cc4", "grid"):
            x = ok_matrix(aok, f"{m}@{k}").astype(bool)
            S["lost_of_full_correct"][f"{m}@{k}"] = [int((fm & ~x).sum()), int((~fm & x).sum())]
    summary["aokvqa"] = S
    md.append(f"## A-OKVQA val (n={n}), restricted-letter accuracy %, [95% bootstrap CI]\n")
    md.append("| Method | " + " | ".join(f"k={k} (R={R_OF[k]})" for k in BUDGETS) + " |")
    md.append("|---|" + "---|" * len(BUDGETS))
    md.append(f"| Full 576 | {S['acc']['full']:.1f} [{S['ci']['full'][0]:.1f}, {S['ci']['full'][1]:.1f}] |" + " |" * (len(BUDGETS) - 1))
    for m in MAIN:
        md.append(f"| {LABEL[m]} | " + " | ".join(
            f"{S['acc'][f'{m}@{k}']:.1f} [{S['ci'][f'{m}@{k}'][0]:.1f}, {S['ci'][f'{m}@{k}'][1]:.1f}]" for k in BUDGETS) + " |")
    md.append("\nPaired difference vs FastV (points), [95% CI], McNemar p\n")
    md.append("| Method | " + " | ".join(f"k={k}" for k in BUDGETS) + " |")
    md.append("|---|" + "---|" * len(BUDGETS))
    for m in MAIN[1:]:
        cells = []
        for k in BUDGETS:
            e = S["vs_fastv"][f"{m}@{k}"]
            c = f"{e['diff']:+.1f} [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]"
            if "p_mcnemar" in e:
                c += f" p={e['p_mcnemar']:.3f}"
            cells.append(c)
        md.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    md.append("\nCC-FastV 4x4 minus attention-free controls (paired, points [95% CI]): " + "; ".join(
        f"{k2}: {v['diff']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]" for k2, v in S["cc4_vs"].items()))
    md.append("\nQuestions answered correctly without pruning but wrong after pruning / wrong without pruning but right after: " + "; ".join(
        f"{k2}: {v[0]}/{v[1]}" for k2, v in S["lost_of_full_correct"].items()))
    md.append("\nGreedy first-token accuracy (official FastV metric: gold letter in decoded first token)\n")
    md.append("| Method | " + " | ".join(f"k={k}" for k in BUDGETS) + " |")
    md.append("|---|" + "---|" * len(BUDGETS))
    md.append(f"| Full 576 | {S['acc_greedy']['full']:.1f} |" + " |" * (len(BUDGETS) - 1))
    for m in ["fastv", "cc4", "grid", "fastv_avgq"]:
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{S['acc_greedy'][f'{m}@{k}']:.1f}" for k in BUDGETS) + " |")
    if "rerun_pred_agreement" in summary:
        md.append("\nRe-run on a second VM, share of identical predictions: " + ", ".join(f"{k2} {100 * v:.1f}%" for k2, v in summary["rerun_pred_agreement"].items()))

# ---------------- MME ----------------
PERC = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
COG = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def mme_scores(recs, name):
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if name.startswith("rand@"):
            k = name.split("@")[1]
            oks = [r[f"rand{s}@{k}"]["ok"] for s in range(3) if f"rand{s}@{k}" in r]
            by[r["category"]][r["qid"]].append(oks)
        else:
            by[r["category"]][r["qid"]].append([r[name]["ok"]])
    out = {}
    for cat, imgs in by.items():
        vals = []
        for s in range(len(next(iter(imgs.values()))[0])):
            q = [o[s] for v in imgs.values() for o in v]
            plus = [all(o[s] for o in v) for v in imgs.values()]
            vals.append(100 * np.mean(q) + 100 * np.mean(plus))
        out[cat] = float(np.mean(vals))
    out["perception"] = sum(out.get(c, 0) for c in PERC)
    out["cognition"] = sum(out.get(c, 0) for c in COG)
    return out


mme = load("mme")
if mme:
    S = {"n": len(mme), "scores": {}}
    S["scores"]["full"] = mme_scores(mme, "full")
    MME_M = [m for m in MAIN if (f"{m}@288" in mme[0] or (m == "rand" and "rand0@288" in mme[0]))]
    for k in BUDGETS:
        for m in MME_M:
            S["scores"][f"{m}@{k}"] = mme_scores(mme, names_for(m, k))
    # paired per-question accuracy diff vs fastv on perception subset (bootstrap over questions)
    S["vs_fastv_perc_acc"] = {}
    perc_idx = [i for i, r in enumerate(mme) if r["category"] in PERC]
    for k in BUDGETS:
        f = ok_matrix(mme, f"fastv@{k}")[perc_idx]
        for m in MME_M[1:]:
            x = ok_matrix(mme, names_for(m, k))[perc_idx]
            lo, hi = boot_ci(x - f)
            S["vs_fastv_perc_acc"][f"{m}@{k}"] = dict(diff=100 * (x - f).mean(), ci=[100 * lo, 100 * hi])
    summary["mme"] = S
    md.append(f"\n## MME (n={len(mme)}): Perception / Cognition score (acc + acc+ summed over subtasks)\n")
    md.append("| Method | " + " | ".join(f"k={k}" for k in BUDGETS) + " |")
    md.append("|---|" + "---|" * len(BUDGETS))
    fs = S["scores"]["full"]
    md.append(f"| Full 576 | {fs['perception']:.0f} / {fs['cognition']:.0f} |" + " |" * (len(BUDGETS) - 1))
    for m in MME_M:
        md.append(f"| {LABEL[m]} | " + " | ".join(
            f"{S['scores'][f'{m}@{k}']['perception']:.0f} / {S['scores'][f'{m}@{k}']['cognition']:.0f}" for k in BUDGETS) + " |")
    md.append("\nMME perception subtasks at k=58 (R=90%) and k=29 (R=95%)\n")
    cats = PERC
    md.append("| Method | " + " | ".join(cats) + " |")
    md.append("|---|" + "---|" * len(cats))
    md.append("| Full 576 | " + " | ".join(f"{fs[c]:.0f}" for c in cats) + " |")
    for k in (58, 29):
        for m in ["fastv", "cc4", "grid", "rand"]:
            sc = S["scores"][f"{m}@{k}"]
            md.append(f"| {m}@{k} | " + " | ".join(f"{sc[c]:.0f}" for c in cats) + " |")

# ---------------- latency ----------------
lat_p = f"{RES}/latency_aokvqa.json"
if os.path.exists(lat_p):
    L = json.load(open(lat_p))
    summary["latency"] = L["rows"]
    md.append(f"\n## Prefill latency (vision encoder + LLM prefill, batch 1, n={L['rows'][0]['n']} A-OKVQA questions, {L['meta']['gpu']})\n")
    md.append("| Method | k | median ms | p10-p90 ms | peak mem GiB |")
    md.append("|---|---|---|---|---|")
    for r in L["rows"]:
        md.append(f"| {r['method']} | {r['k']} | {r['median_ms']:.1f} | {r['p10_ms']:.1f}-{r['p90_ms']:.1f} | {r['peak_mem_gib']:.2f} |")

# ---------------- theoretical FLOPs (FastV Eq. 5, LLM only) ----------------
d, mff, T, K = 4096, 11008, 32, 2
n_text = 0
if aok:
    meta = json.load(open(f"{RES}/meta_aokvqa.json"))
n_seq = summary.get("seq_len_mean", None)


def flops_layer(n):
    return 4 * n * d * d + 2 * n * n * d + 2 * n * d * mff  # FastV's estimate (attention + FFN)


fl = {}
n_full = 576 + 100  # fallback if sequence lengths were not recorded
if os.path.exists(NR):
    n_full = int(round(np.mean([json.loads(l)["seq_len"] for l in open(NR)])))
summary["seq_len_mean"] = n_full
for k in [576] + BUDGETS:
    nk = n_full - 576 + k
    fl[k] = (K * flops_layer(n_full) + (T - K) * flops_layer(nk)) / (T * flops_layer(n_full))
summary["flops_ratio_nominal"] = fl
md.append(f"\n## Theoretical LLM FLOPs ratio (FastV Eq.5 with mean sequence length n={n_full} tokens, K=2)\n")
md.append("| k | " + " | ".join(str(k) for k in [576] + BUDGETS) + " |")
md.append("|---|" + "---|" * 5)
md.append("| FLOPs ratio | " + " | ".join(f"{100 * fl[k]:.0f}%" for k in [576] + BUDGETS) + " |")

# ---------------- spatial analysis + figures ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
SHORT = {"fastv": "FastV", "cc4": "CC-FastV 4x4 (proposed)", "cc2": "CC-FastV 2x2", "grid": "Uniform grid", "rand": "Random",
         "fastv_avgq": "Paper-text criterion", "fastvnr": "RoPE-free score"}
COL = {"fastv": "C0", "cc4": "C1", "cc2": "C2", "grid": "C3", "rand": "C4", "rand0": "C4", "fastv_avgq": "C5", "fastvnr": "C6"}

sel_p = f"{RES}/selection_aokvqa.npz"
if os.path.exists(sel_p):
    Z = dict(np.load(sel_p))
    if os.path.exists(f"{RES}/norope/selection_aokvqa.npz"):
        Zn = np.load(f"{RES}/norope/selection_aokvqa.npz")
        for k in BUDGETS:
            Z[f"fastvnr_at_{k}"] = Zn[f"fastvnr_at_{k}"]
    nq = summary["aokvqa"]["n"]
    SPM = ["fastv", "fastv_avgq", "cc4", "cc2", "grid", "rand0"] + (["fastvnr"] if "fastvnr_at_58" in Z else [])
    sp = {}
    for k in BUDGETS:
        for m in SPM:
            c = Z[f"{m}_at_{k}"].reshape(24, 24)
            rows = c.sum(1)
            ybar = float((rows * np.arange(24)).sum() / rows.sum() / 23)
            top_half = float(rows[:12].sum() / rows.sum())
            # mean number of distinct 4x4 blocks (of 16) occupied is not recoverable from counts; report row stats
            sp[f"{m}@{k}"] = dict(mean_y=ybar, frac_top_half=top_half)
    from scipy.stats import chisquare
    for k in BUDGETS:  # chi-square test of kept-token rows against a uniform row distribution (as in FEATHER)
        rows_k = Z[f"fastv_at_{k}"].reshape(24, 24).sum(1)
        sp[f"chi2_fastv@{k}"] = float(chisquare(rows_k).pvalue)
    am = Z["attn_mean"].reshape(24, 24)
    sp["attn_mass_top_half"] = float(am[:12].sum() / am.sum())
    sp["attn_mass_bottom_row"] = float(am[23].sum() / am.sum())
    summary["spatial"] = sp
    md.append("\n## Spatial distribution of kept tokens on A-OKVQA (mean normalised row: 0=top, 1=bottom; share kept in top half)\n")
    md.append("| Method | " + " | ".join(f"k={k}" for k in BUDGETS) + " |")
    md.append("|---|" + "---|" * len(BUDGETS))
    for m in SPM:
        md.append(f"| {m} | " + " | ".join(f"{sp[f'{m}@{k}']['mean_y']:.2f} / {100 * sp[f'{m}@{k}']['frac_top_half']:.0f}%" for k in BUDGETS) + " |")
    md.append(f"\nLast-token attention mass (layer 1, head-mean, per-image normalised) in top half: {100 * sp['attn_mass_top_half']:.1f}%; in last row: {100 * sp['attn_mass_bottom_row']:.1f}%")

    panels = [("attn_mean", "Mean last-token attn.\n(layer 1)"), ("fastv_at_58", "FastV (code)\nkeep freq., k=58"),
              ("fastv_avgq_at_58", "Paper-text criterion\nkeep freq., k=58")]
    if "fastvnr_at_58" in Z:
        panels.append(("fastvnr_at_58", "RoPE-free score\nkeep freq., k=58"))
    panels.append(("cc4_at_58", "CC-FastV 4x4\nkeep freq., k=58"))
    fig, ax = plt.subplots(1, len(panels), figsize=(7.6, 1.75))
    for a, (key, title) in zip(ax, panels):
        im = a.imshow(Z[key].reshape(24, 24) / (1 if key == "attn_mean" else nq), cmap="viridis")
        a.set_title(title, fontsize=6.5)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(f"{FIG}/spatial_maps.png", dpi=220)
    plt.close()

    fig, a = plt.subplots(figsize=(3.4, 2.3))
    for m, st in [("fastv", "-o"), ("fastv_avgq", "-^"), ("cc4", "-s"), ("grid", "--"), ("rand0", ":")] + ([("fastvnr", "-.")] if "fastvnr_at_58" in Z else []):
        c = Z[f"{m}_at_58"].reshape(24, 24).sum(1)
        a.plot(np.arange(24), c / c.sum(), st, ms=2.5, lw=1, label=m, color=COL[m])
    a.set_xlabel("image row (0 = top)"); a.set_ylabel("share of kept tokens"); a.set_title("Row distribution of kept tokens, k=58", fontsize=8)
    a.legend(fontsize=6, frameon=False)
    plt.tight_layout(); plt.savefig(f"{FIG}/row_distribution.png", dpi=220); plt.close()

if aok:
    S = summary["aokvqa"]
    fig, axs = plt.subplots(1, 2 if mme else 1, figsize=(7.2 if mme else 3.6, 2.5))
    axs = np.atleast_1d(axs)
    a = axs[0]
    xs = [576] + BUDGETS
    for m, st in [("fastv", "-o"), ("cc4", "-s"), ("cc2", "-d"), ("grid", "--x"), ("rand", ":v"), ("fastv_avgq", "-^")] + ([("fastvnr", "-.p")] if "fastvnr" in MAIN else []):
        ys = [S["acc"]["full"]] + [S["acc"][f"{m}@{k}"] for k in BUDGETS]
        a.plot(xs, ys, st, ms=3, lw=1, label=SHORT[m], color=COL[m])
    a.set_xscale("log"); a.set_xticks(xs); a.set_xticklabels([str(x) for x in xs])
    a.set_xlabel("image tokens kept after layer 2 (k)"); a.set_ylabel("accuracy (%)"); a.set_title("A-OKVQA val (MC)", fontsize=8)
    a.legend(fontsize=5.2, frameon=False, loc="lower right")
    if mme:
        a = axs[1]
        SM = summary["mme"]["scores"]
        for m, st in [("fastv", "-o"), ("cc4", "-s"), ("grid", "--x"), ("rand", ":v")]:
            ys = [SM["full"]["perception"]] + [SM[f"{m}@{k}"]["perception"] for k in BUDGETS]
            a.plot(xs, ys, st, ms=3, lw=1, label=SHORT[m], color=COL[m])
        a.legend(fontsize=5.2, frameon=False, loc="lower right")
        a.set_xscale("log"); a.set_xticks(xs); a.set_xticklabels([str(x) for x in xs])
        a.set_xlabel("image tokens kept after layer 2 (k)"); a.set_ylabel("perception score"); a.set_title("MME perception", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/accuracy_vs_budget.png", dpi=220); plt.close()

json.dump(summary, open(f"{RES}/summary.json", "w"), indent=1)
open(f"{RES}/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
