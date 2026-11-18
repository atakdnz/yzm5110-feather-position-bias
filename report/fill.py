"""Inserts the result tables (generated from results/summary.json) into report.html.
Writes report/report_filled.html, which scripts/build_pdf.sh renders. No number in the
tables is typed by hand."""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{ROOT}/results/summary.json"))
B = [288, 144, 58, 29]
R = {288: "50%", 144: "75%", 58: "90%", 29: "95%"}
LAB = {"fastv": "FastV (official code: last-token attention)", "fastv_avgq": "FastV, paper-text criterion (mean received attention)",
       "cc4": "<b>CC-FastV 4&times;4 (proposed)</b>", "cc2": "CC-FastV 2&times;2 (ablation)", "grid": "Uniform grid (no attention)",
       "rand": "Random (mean of 3 seeds)", "fastvnr": "FastV, RoPE-free score (FEATHER &phi;<sub>-R</sub>)"}
A = S["aokvqa"]


def f1(x):
    return f"{x:.1f}"


def ci(key):
    lo, hi = A["ci"][key]
    return f"{A['acc'][key]:.1f} <span class='small'>[{lo:.1f}, {hi:.1f}]</span>"


def table(caption, head, rows, cls=""):
    h = "".join(f"<th{' class=num' if i else ''}>{x}</th>" for i, x in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table class='{cls}'><caption>{caption}</caption><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


T = {}
# ---- reproduction table: paper vs measured ----
paper = {"full": 76.7, 288: 77.0, 144: 75.5, 58: 70.1}
paper_rand_k0 = {288: 75.3, 144: 72.8, 58: 66.8}
rows = [["No pruning (576 tokens)", "76.7", f1(A["acc_greedy"]["full"]), ci("full")]]
for k in B:
    rows.append([f"FastV <i>K</i>=2, <i>R</i>={R[k]} (<i>k</i>={k})", f1(paper[k]) if k in paper else "&ndash;",
                 f1(A["acc_greedy"][f"fastv@{k}"]), ci(f"fastv@{k}")])
for k in B[:3]:
    rows.append([f"Random dropping, <i>R</i>={R[k]}", f"{paper_rand_k0[k]:.1f} (<i>K</i>=0)", "&ndash;", ci(f"rand@{k}") + " (<i>K</i>=2)"])
T["TABLE_REPRO"] = table(
    "<b>Table 3.</b> Baseline sanity check (FastV). Left: values reported by the authors for LLaVA-1.5-7B on A-OKVQA [1, Table 1] "
    "<span class='tag auth'>AUTHORS</span>. Right: measurements of this report <span class='tag ours'>MEASURED</span> "
    "(official metric = gold letter in the greedy first token; restricted = arg-max over the option letters, with 95% bootstrap CI). "
    "The paper's random rows prune before the LLM (<i>K</i>=0); mine prune at <i>K</i>=2, so those rows are not directly comparable.",
    ["Setting", "Paper (reported)", "Measured, official metric", "Measured, restricted [95% CI]"], rows)

# ---- main comparison ----
methods = ["fastv", "cc4", "cc2", "grid", "rand", "fastv_avgq"] + (["fastvnr"] if "fastvnr@288" in A["acc"] else [])
rows = [["No pruning (576)"] + [ci("full")] + [""] * 3]
for m in methods:
    rows.append([LAB[m]] + [ci(f"{m}@{k}") for k in B])
T["TABLE_MAIN"] = table(
    f"<b>Table 5.</b> A-OKVQA val accuracy (%, restricted metric, n={A['n']}) for every selection rule at equal budgets "
    "<span class='tag ours'>MEASURED</span>. All rules keep the same number of image tokens after layer 2, so their FLOPs are identical within a column. "
    "Brackets: 95% bootstrap CI. Random seeds individually: " +
    "; ".join(f"<i>k</i>={k}: " + "/".join(f1(x) for x in A["acc"][f"rand_seeds@{k}"]) for k in B) + ".",
    ["Selection rule"] + [f"<i>k</i>={k} (<i>R</i>={R[k]})" for k in B], rows)

rows = []
for m in methods[1:]:
    cells = []
    for k in B:
        e = A["vs_fastv"][f"{m}@{k}"]
        c = f"{e['diff']:+.1f} <span class='small'>[{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]</span>"
        if "p_mcnemar" in e:
            c += f"<br><span class='small'>{'p&lt;0.001' if e['p_mcnemar'] < 0.001 else 'p=%.3f' % e['p_mcnemar']}; {e['fastv_wrong_m_right']}&uarr;/{e['fastv_right_m_wrong']}&darr;</span>"
        cells.append(c)
    rows.append([LAB[m]] + cells)
T["TABLE_DIFF"] = table(
    "<b>Table 6.</b> Paired accuracy difference to FastV on the same questions (percentage points, 95% paired-bootstrap CI), exact McNemar p-value, "
    "and the discordant counts (&uarr; = questions FastV gets wrong and the rule gets right; &darr; = the reverse) <span class='tag ours'>MEASURED</span>.",
    ["Rule minus FastV"] + [f"<i>k</i>={k}" for k in B], rows)

# ---- MME ----
if "mme" in S:
    M = S["mme"]["scores"]
    mm = [m for m in ["fastv", "cc4", "grid", "rand"] if f"{m}@288" in M]
    rows = [["No pruning (576)", f"{M['full']['perception']:.0f} / {M['full']['cognition']:.0f}", "", "", ""]]
    for m in mm:
        rows.append([LAB[m].replace(" (mean of 3 seeds)", " (1 seed)")] + [f"{M[f'{m}@{k}']['perception']:.0f} / {M[f'{m}@{k}']['cognition']:.0f}" for k in B])
    T["TABLE_MME"] = table(
        f"<b>Table 7.</b> MME (n={S['mme']['n']}) Perception / Cognition scores (sum over subtasks of accuracy + accuracy+; maxima 2000 / 800) "
        "<span class='tag ours'>MEASURED</span>.", ["Selection rule"] + [f"<i>k</i>={k}" for k in B], rows)
    cats = ["existence", "count", "position", "color", "OCR", "posters", "celebrity", "scene", "landmark", "artwork"]
    rows = [["No pruning"] + [f"{M['full'][c]:.0f}" for c in cats]]
    for k in (58, 29):
        for m in mm:
            rows.append([f"{m.replace('rand', 'random')} @{k}"] + [f"{M[f'{m}@{k}'][c]:.0f}" for c in cats])
    T["TABLE_MME_SUB"] = table(
        "<b>Table 8.</b> MME perception subtasks (score = acc + acc+, max 200) at <i>k</i>=58 and <i>k</i>=29 <span class='tag ours'>MEASURED</span>.",
        ["Rule @k"] + [c.capitalize() if c != "OCR" else c for c in cats], rows)
    d = S["mme"]["vs_fastv_perc_acc"]
    T["MME_DIFF_TEXT"] = "; ".join(
        f"<i>k</i>={k}: CC-FastV {d[f'cc4@{k}']['diff']:+.1f} [{d[f'cc4@{k}']['ci'][0]:+.1f}, {d[f'cc4@{k}']['ci'][1]:+.1f}]" for k in B)

# ---- spatial ----
sp = S["spatial"]
spm = [m for m in ["fastv", "fastv_avgq", "fastvnr", "cc4", "cc2", "grid", "rand0"] if f"{m}@288" in sp]
rows = []
for m in spm:
    rows.append([LAB.get(m, m).replace("<b>", "").replace("</b>", "") if m != "rand0" else "Random (seed 0)"] +
                [f"{sp[f'{m}@{k}']['mean_y']:.2f} / {100 * sp[f'{m}@{k}']['frac_top_half']:.0f}%" for k in B])
T["TABLE_SPATIAL"] = table(
    "<b>Table 9.</b> Where the kept tokens are on A-OKVQA: mean normalised row of kept tokens (0 = top row, 1 = bottom row; 0.50 = balanced) / share of kept tokens in the top half of the 24&times;24 grid "
    "<span class='tag ours'>MEASURED</span>.", ["Selection rule"] + [f"<i>k</i>={k}" for k in B], rows)

# ---- latency ----
if "latency" in S:
    L = {(r["method"], r["k"]): r for r in S["latency"]}
    fl = S["flops_ratio_nominal"]
    full = L[("full", 576)]
    def cell(m, k):
        r = L[(m, k)]
        return f"{r['median_ms']:.1f} <span class='small'>({r['p10_ms']:.1f}&ndash;{r['p90_ms']:.1f})</span>"
    rows = [["576 (no pruning)", "100%", cell("full", 576), "", "", "", f"{full['peak_mem_gib']:.2f}"]]
    for k in B:
        rows.append([f"{k} (<i>R</i>={R[k]})", f"{100 * fl[str(k)]:.0f}%"] + [cell(m, k) for m in ["fastv", "cc4", "grid", "rand0"]] +
                    [f"{L[('fastv', k)]['peak_mem_gib']:.2f} / {L[('grid', k)]['peak_mem_gib']:.2f}"])
    T["TABLE_LAT"] = table(
        f"<b>Table 10.</b> Prefill latency per question in ms, median (10th&ndash;90th percentile) over {full['n']} A-OKVQA questions: vision encoder + projector + LLM prefill + selection, batch 1, idle A100-40GB. "
        f"FLOPs = FastV's theoretical LLM FLOPs ratio for the mean sequence length of {S.get('seq_len_mean', '?')} tokens. Peak = peak allocated GiB (with / without the attention-scoring pass) <span class='tag ours'>MEASURED</span>.",
        ["Image tokens <i>k</i>", "FLOPs", "FastV", "CC-FastV 4&times;4", "Uniform grid", "Random", "Peak GiB"], rows)

html = open(f"{ROOT}/report/report.html").read()


def token(m):
    kind, key = m.group(1), m.group(2)
    if kind == "ACC":
        return f"{A['acc'][key]:.1f}"
    if kind == "VS":
        e = A["vs_fastv"][key]
        return f"{e['diff']:+.1f} [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]"
    if kind == "VSD":
        return f"{A['vs_fastv'][key]['diff']:+.1f}"
    if kind == "P":
        p = A["vs_fastv"][key]["p_mcnemar"]
        return "p&lt;0.001" if p < 0.001 else f"p={p:.3f}"
    if kind == "UP":
        return str(A["vs_fastv"][key]["fastv_wrong_m_right"])
    if kind == "DOWN":
        return str(A["vs_fastv"][key]["fastv_right_m_wrong"])
    if kind == "CV":
        e = A["cc4_vs"][key]
        return f"{e['diff']:+.1f} [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]"
    if kind == "MY":
        return f"{S['spatial'][key]['mean_y']:.2f}"
    if kind == "TOP":
        return f"{100 * S['spatial'][key]['frac_top_half']:.0f}%"
    if kind == "MP":
        return f"{S['mme']['scores'][key]['perception']:.0f}"
    if kind == "MD":
        e = S["mme"]["vs_fastv_perc_acc"][key]
        return f"{e['diff']:+.1f} [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]"
    if kind == "MPD":
        M = S["mme"]["scores"]
        return f"{M[f'cc4@{key}']['perception'] - M[f'fastv@{key}']['perception']:+.0f}"
    if kind == "CHI":
        p = S["spatial"][key]
        return "p&lt;10<sup>&minus;10</sup>" if p < 1e-10 else f"p={p:.2g}"
    raise KeyError(kind)


html = re.sub(r"\[\[([A-Z]+):([^\]]+)\]\]", token, html)
for key, val in T.items():
    html = html.replace(f"<!--{key}-->", val)
left = re.findall(r"<!--(TABLE_[A-Z_]+|MME_DIFF_TEXT)-->", html)
if left:
    print("unfilled markers:", left)
open(f"{ROOT}/report/report_filled.html", "w").write(html)
print("filled", list(T))
