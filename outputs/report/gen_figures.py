#!/usr/bin/env python3
"""Generate all report figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':                  'serif',
    'font.serif':                   ['cmr10', 'DejaVu Serif'],
    'font.size':                    10,
    'axes.formatter.use_mathtext':  True,
    'axes.unicode_minus':           False,
    'axes.spines.top':              False,
    'axes.spines.right':            False,
    'axes.grid':                    True,
    'grid.alpha':                   0.25,
    'grid.linewidth':               0.5,
    'figure.dpi':                   150,
    'savefig.bbox':                 'tight',
    'savefig.pad_inches':           0.1,
})

# Paul Tol "bright" — colorblind-safe academic palette
C1 = '#4477AA'   # blue
C2 = '#EE6677'   # red
C3 = '#228833'   # green
C4 = '#CCBB44'   # yellow / olive
C5 = '#66CCEE'   # cyan
C6 = '#AA3377'   # purple

# ── data ──────────────────────────────────────────────────────────────────
RAW = {
    'MB-E1':     {'D1':[.8060,.8190,.7984,.8201,.8125],'D2':[.4159,.2544,.3193,.3266,.2886],'D3':[.6263,.6604,.6566,.6459,.6610],'D4':[.7138,.7295,.7180,.7119,.7311]},
    'E1':        {'D1':[.8469,.8539,.8246,.8288,.8347],'D2':[.6193,.5774,.4468,.4849,.6152],'D3':[.7436,.7746,.7531,.7675,.7461],'D4':[.7674,.7819,.7699,.7708,.7756]},
    'E2':        {'D1':[.8402,.8326,.8248,.8512,.8388],'D2':[.5510,.4859,.5373,.5056,.5215],'D3':[.7672,.7953,.7651,.7701,.8018],'D4':[.7720,.7905,.7826,.7868,.7853]},
    'E3':        {'D1':[.8519,.8346,.8009,.8500,.8292],'D2':[.5145,.5308,.4959,.5056,.5675],'D3':[.7428,.7753,.7724,.7546,.7724],'D4':[.7586,.7877,.7714,.7716,.7818]},
    'E4':        {'D1':[.1408,.1813,.1842,.2411,.1561],'D2':[.2208,.1303,.0991,.1318,.1977],'D3':[.5031,.4875,.4751,.4778,.4821],'D4':[.5709,.5590,.5307,.5567,.5617]},
    'E5':        {'D1':[.2965,.2584,.3068,.3310,.2867],'D2':[.2372,.2037,.2028,.2012,.2323],'D3':[.6200,.6054,.5740,.6066,.5753],'D4':[.6360,.6179,.6104,.6399,.6424]},
    'IR-E1':     {'D1':[.7986,.8370,.7856,.8168,.8013],'D2':[.3155,.3293,.3795,.2825,.2954],'D3':[.7458,.7632,.7549,.7568,.7412],'D4':[.7510,.7561,.7807,.7406,.7591]},
    'IR-E2':     {'D1':[.8065,.8110,.8128,.8356,.8412],'D2':[.3294,.4214,.3858,.2963,.3296],'D3':[.7505,.7712,.7629,.7665,.7404],'D4':[.7688,.7583,.7776,.7811,.7698]},
    'IR-E3':     {'D1':[.8061,.8520,.8047,.8128,.8099],'D2':[.2646,.3044,.3874,.3317,.3247],'D3':[.7662,.7636,.7273,.7657,.7522],'D4':[.7485,.7788,.7850,.7581,.7670]},
    'IR-E2-6ep': {'D1':[.8122,.8451,.8123,.8457,.8297],'D2':[.4160,.3494,.3837,.4233,.4347],'D3':[.7473,.7576,.7415,.7591,.7525],'D4':[.7612,.7666,.7799,.7666,.7749]},
    'IR-E3-6ep': {'D1':[.8144,.8310,.7947,.8216,.8369],'D2':[.2916,.3371,.4000,.3181,.3532],'D3':[.7508,.7715,.7420,.7836,.7435],'D4':[.7453,.7577,.7736,.7697,.7750]},
}
DS   = ['D1', 'D2', 'D3', 'D4']
DS_L = ['D1 - EmoTweetID\n(6 classes)', 'D2 - EmoCMT\n(5 classes)',
        'D3 - EmoT\n(5 classes)',        'D4 - EmoPubID\n(6 classes)']

def signed_fmt(x, _): return f'{"+" if x >= 0 else "-"}{abs(x):.3f}'

def m(exp, ds):  return np.mean(RAW[exp][ds])
def s(exp, ds):  return np.std(RAW[exp][ds], ddof=1)
def means(exp):  return [m(exp, d) for d in DS]
def stds(exp):   return [s(exp, d) for d in DS]
def delta(a, b): return [m(b, d) - m(a, d) for d in DS]
def avg(exp):    return np.mean(means(exp))

CURVES = {
    'xlmt-full':   [(0.156,3.1444),(0.312,3.0576),(0.467,3.0096),(0.623,2.9467),(0.779,2.9033),
                    (0.935,2.8957),(1.09,2.8387),(1.246,2.7896),(1.402,2.7440),(1.558,2.7498),
                    (1.714,2.7459),(1.869,2.7123),(2.025,2.7058),(2.181,2.6629),(2.337,2.6694),
                    (2.493,2.6465),(2.648,2.6217),(2.804,2.6309),(2.960,2.6026)],
    'xlmt-emb':    [(0.156,3.1335),(0.312,3.0767),(0.467,3.0558),(0.623,3.0124),(0.779,3.0052),
                    (0.935,3.0117),(1.09,2.9809),(1.246,2.9478),(1.402,2.9131),(1.558,2.9381),
                    (1.714,2.9500),(1.869,2.9304)],
    'ir-full-3ep': [(0.156,3.7481),(0.312,3.6236),(0.467,3.5028),(0.623,3.4678),(0.779,3.4074),
                    (0.935,3.3771),(1.09,3.3478),(1.246,3.2790),(1.402,3.2815),(1.558,3.2389),
                    (1.714,3.2229),(1.869,3.1937),(2.025,3.1786),(2.181,3.1440),(2.337,3.1378),
                    (2.493,3.1254),(2.648,3.1021),(2.804,3.0860),(2.960,3.0824)],
    'ir-full-6ep': [(0.156,3.7805),(0.467,3.5696),(0.779,3.4693),(1.09,3.4158),(1.402,3.3398),
                    (1.714,3.2873),(2.025,3.2509),(2.337,3.2107),(2.648,3.1683),(2.96,3.1406),
                    (3.271,3.1161),(3.583,3.0843),(3.895,3.0512),(4.206,3.0499),(4.518,3.0175),
                    (4.829,3.0127),(5.141,2.9788),(5.452,2.9749),(5.920,2.9409)],
    'ir-emb-3ep':  [(0.156,4.0858),(0.467,3.8125),(0.779,3.7233),(1.09,3.6849),(1.402,3.6432),
                    (1.714,3.6122),(2.025,3.5937),(2.337,3.5818),(2.648,3.5672),(2.960,3.5562)],
    'ir-emb-6ep':  [(0.156,4.0862),(0.467,3.8092),(0.779,3.7169),(1.09,3.6744),(1.402,3.6286),
                    (1.714,3.5926),(2.025,3.5706),(2.337,3.5502),(2.648,3.5273),(2.960,3.5058),
                    (3.271,3.4975),(3.583,3.4802),(3.895,3.4684),(4.206,3.4780),(4.518,3.4506),
                    (4.829,3.4695),(5.141,3.4458),(5.297,3.4411)],
}

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — DAPT Loss Curves
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

ax = axes[0]
for key, lbl, color, ls in [
    ('xlmt-full', 'A-full (100% param)', C1, '-'),
    ('xlmt-emb',  'A-emb (69.4% param)', C2, '--'),
]:
    ep, lss = zip(*CURVES[key])
    ax.plot(ep, lss, color=color, ls=ls, lw=1.8, marker='o', ms=2.5, label=lbl)
    ax.plot(ep[-1], lss[-1], marker='*', ms=8, color=color, zorder=5)

ax.set_xlabel('Epoch')
ax.set_ylabel('Eval Loss (MLM)')
ax.set_title('XLM-T DAPT')
ax.set_xlim(0, 3.2)
ax.legend(loc='upper right')
ax.annotate('early stop\n(ep 1.87)', xy=(1.869, 2.9304), xytext=(2.1, 3.05),
            arrowprops=dict(arrowstyle='->', color=C2, lw=1), color=C2)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

ax = axes[1]
for key, lbl, color, ls, alpha in [
    ('ir-full-6ep', 'A-full 6ep (best)',           C1, '-',  1.0),
    ('ir-full-3ep', 'A-full 3ep',                   C1, '--', 0.55),
    ('ir-emb-6ep',  'A-emb 6ep (early stop ep 5.3)', C2, '-',  1.0),
    ('ir-emb-3ep',  'A-emb 3ep',                    C2, '--', 0.55),
]:
    ep, lss = zip(*CURVES[key])
    ax.plot(ep, lss, color=color, ls=ls, lw=1.8, alpha=alpha, marker='o', ms=1.8, label=lbl)

ax.set_xlabel('Epoch')
ax.set_ylabel('Eval Loss (MLM)')
ax.set_title('Indo-RoBERTa DAPT')
ax.set_xlim(0, 6.3)
ax.legend(loc='upper right')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig(OUT / 'fig1_dapt_curves.png')
plt.close()
print("Figure 1 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Fine-tuning Results: XLM-T
# ══════════════════════════════════════════════════════════════════
xlmt_exps   = ['E1', 'E2', 'E3', 'E4', 'E5']
xlmt_lbls   = ['No DAPT', 'A-full DAPT', 'A-emb DAPT',
               'A-emb DAPT + freeze FT', 'A-full DAPT + freeze FT']
xlmt_colors = [C1, C2, C3, C5, C6]

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=False)
axes = axes.flatten()
for j, (dskey, ax) in enumerate(zip(DS, axes)):
    ms = [m(e, dskey) for e in xlmt_exps]
    ss = [s(e, dskey) for e in xlmt_exps]
    y  = np.arange(len(xlmt_exps))
    ax.barh(y, ms, color=xlmt_colors, alpha=0.85, height=0.6, zorder=3)
    ax.errorbar(ms, y, xerr=ss, fmt='none', color='#555', capsize=3, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(xlmt_lbls)
    ax.invert_yaxis()
    ax.set_title(DS_L[j])
    ax.set_xlabel('Macro-F1')
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.axvline(ms[0], color=C1, lw=0.8, ls=':', alpha=0.7)

plt.tight_layout()
plt.savefig(OUT / 'fig2_xlmt_results.png')
plt.close()
print("Figure 2 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Fine-tuning Results: Indo-RoBERTa
# ══════════════════════════════════════════════════════════════════
ir_exps   = ['IR-E1', 'IR-E2', 'IR-E3', 'IR-E2-6ep', 'IR-E3-6ep']
ir_lbls   = ['No DAPT', 'A-full DAPT 3ep', 'A-emb DAPT 3ep',
             'A-full DAPT 6ep', 'A-emb DAPT 6ep']
ir_colors = [C1, C2, C3, C4, C5]

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=False)
axes = axes.flatten()
for j, (dskey, ax) in enumerate(zip(DS, axes)):
    ms = [m(e, dskey) for e in ir_exps]
    ss = [s(e, dskey) for e in ir_exps]
    y  = np.arange(len(ir_exps))
    ax.barh(y, ms, color=ir_colors, alpha=0.85, height=0.6, zorder=3)
    ax.errorbar(ms, y, xerr=ss, fmt='none', color='#555', capsize=3, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ir_lbls)
    ax.invert_yaxis()
    ax.set_title(DS_L[j])
    ax.set_xlabel('Macro-F1')
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.axvline(ms[0], color=C1, lw=0.8, ls=':', alpha=0.7)

plt.tight_layout()
plt.savefig(OUT / 'fig3_ir_results.png')
plt.close()
print("Figure 3 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — RQ1: Δ DAPT Gain
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

x, w = np.arange(4), 0.35

ax = axes[0]
d_full = delta('E1', 'E2')
d_emb  = delta('E1', 'E3')
ax.bar(x - w/2, d_full, w, label='A-full (E2 - E1)', color=C2, alpha=0.85)
ax.bar(x + w/2, d_emb,  w, label='A-emb  (E3 - E1)', color=C3, alpha=0.85)
ax.axhline(0, color='#555', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(DS, rotation=90, ha='center')
ax.set_ylabel('Delta Macro-F1')
ax.set_title('XLM-T: DAPT Gain vs. No-DAPT')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(signed_fmt))

ax = axes[1]
d_full6 = delta('IR-E1', 'IR-E2-6ep')
d_emb6  = delta('IR-E1', 'IR-E3-6ep')
ax.bar(x - w/2, d_full6, w, label='A-full 6ep (IR-E2-6ep - IR-E1)', color=C2, alpha=0.85)
ax.bar(x + w/2, d_emb6,  w, label='A-emb 6ep  (IR-E3-6ep - IR-E1)', color=C3, alpha=0.85)
ax.axhline(0, color='#555', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(DS, rotation=90, ha='center')
ax.set_ylabel('Delta Macro-F1')
ax.set_title('Indo-RoBERTa: DAPT Gain vs. No-DAPT')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(signed_fmt))

plt.tight_layout()
plt.savefig(OUT / 'fig4_rq1_dapt_gain.png')
plt.close()
print("Figure 4 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — RQ2: A-emb vs A-full
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 3.8))

x, w   = np.arange(4), 0.25
d_xlmt = delta('E2',       'E3')
d_ir3  = delta('IR-E2',    'IR-E3')
d_ir6  = delta('IR-E2-6ep','IR-E3-6ep')

ax.bar(x - w,  d_xlmt, w, label='XLM-T (E3 - E2)',                        color=C1, alpha=0.85)
ax.bar(x,      d_ir3,  w, label='Indo-RoBERTa 3ep (IR-E3 - IR-E2)',        color=C3, alpha=0.85)
ax.bar(x + w,  d_ir6,  w, label='Indo-RoBERTa 6ep (IR-E3-6ep - IR-E2-6ep)', color=C4, alpha=0.85)
ax.axhline(0, color='#555', lw=1.0, ls='--', label='No difference')
ax.set_xticks(x); ax.set_xticklabels(DS, rotation=90, ha='center')
ax.set_ylabel('Delta Macro-F1  (A-emb - A-full)')
ax.set_title('RQ2: A-emb vs. A-full DAPT (positive = A-emb better)')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(signed_fmt))

plt.tight_layout()
plt.savefig(OUT / 'fig5_rq2_emb_vs_full.png')
plt.close()
print("Figure 5 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Efficiency Trade-off
# ══════════════════════════════════════════════════════════════════
EFF = [
    # label,            model,   %param,  minutes, avg_f1,           ppl
    ('XLM-T A-full',   'xlmt',  100.0,   213.6,   avg('E2'),        13.50),
    ('XLM-T A-emb',    'xlmt',   69.44,  121.4,   avg('E3'),        18.41),
    ('IR A-full 3ep',  'ir',    100.0,    92.2,   avg('IR-E2'),     21.81),
    ('IR A-emb 3ep',   'ir',     31.79,   73.9,   avg('IR-E3'),     34.97),
    ('IR A-full 6ep',  'ir',    100.0,   184.9,   avg('IR-E2-6ep'), 18.93),
    ('IR A-emb 6ep',   'ir',     31.79,  142.9,   avg('IR-E3-6ep'), 31.22),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

ax = axes[0]
for lbl, mdl, pct, mins, f1, ppl in EFF:
    c = C1 if mdl == 'xlmt' else C3
    ax.scatter(pct, mins, s=80 + f1 * 400, color=c, alpha=0.85, edgecolors='#555', lw=0.7, zorder=3)
    ax.annotate(lbl, (pct, mins), textcoords='offset points',
                xytext=(4, -4 if mdl == 'xlmt' else 6))

ax.set_xlabel('% Trainable Parameters (DAPT)')
ax.set_ylabel('DAPT Training Time (min)')
ax.set_title('Time vs. Parameters\n(bubble size = avg downstream Macro-F1)')
ax.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C1, ms=8, label='XLM-T'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C3, ms=8, label='Indo-RoBERTa'),
])

ax = axes[1]
for lbl, mdl, pct, mins, f1, ppl in EFF:
    c = C2 if 'full' in lbl else C3
    ax.scatter(ppl, f1, s=70, color=c, alpha=0.85, edgecolors='#555', lw=0.7, zorder=3)
    ax.annotate(lbl, (ppl, f1), textcoords='offset points', xytext=(3, -6))

ax.set_xlabel('DAPT Eval Perplexity (lower = better)')
ax.set_ylabel('Avg Macro-F1 (downstream)')
ax.set_title('DAPT Perplexity vs. Downstream Performance')
ax.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C2, ms=8, label='A-full'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C3, ms=8, label='A-emb'),
])

plt.tight_layout()
plt.savefig(OUT / 'fig6_efficiency.png')
plt.close()
print("Figure 6 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 7 — Summary Heatmap (11 conditions incl. mBERT)
# ══════════════════════════════════════════════════════════════════
all_exps = ['MB-E1',
            'E1','E2','E3','E4','E5',
            'IR-E1','IR-E2','IR-E3','IR-E2-6ep','IR-E3-6ep']
exp_lbls = ['MB-E1  No DAPT (mBERT)',
            'E1  No DAPT (XLM-T)', 'E2  A-full (XLM-T)', 'E3  A-emb (XLM-T)',
            'E4  A-emb+freeze (XLM-T)', 'E5  A-full+freeze (XLM-T)',
            'IR-E1  No DAPT (IR)', 'IR-E2  A-full 3ep (IR)', 'IR-E3  A-emb 3ep (IR)',
            'IR-E2-6ep  A-full 6ep (IR)', 'IR-E3-6ep  A-emb 6ep (IR)']

data_mat = np.array([[m(e, d) for d in DS] for e in all_exps])

fig, ax = plt.subplots(figsize=(7, 6.5))
im = ax.imshow(data_mat, cmap='Blues', vmin=0.1, vmax=0.95, aspect='auto')
plt.colorbar(im, ax=ax, label='Macro-F1', shrink=0.8)

ax.set_xticks(range(4)); ax.set_xticklabels(DS)
ax.set_yticks(range(11)); ax.set_yticklabels(exp_lbls)
ax.set_xlabel('Dataset')
ax.set_title('Macro-F1 Heatmap — All Experiments $\\times$ Datasets')
ax.axhline(0.5, color='white', lw=2)   # separator: mBERT | XLM-T
ax.axhline(5.5, color='white', lw=2)   # separator: XLM-T | IR
ax.text(3.55, 0.0,  'mBERT', color='#555', va='center', rotation=90, fontsize=8)
ax.text(3.55, 3.0,  'XLM-T', color='#555', va='center', rotation=90, fontsize=8)
ax.text(3.55, 8.0,  'IR',    color='#555', va='center', rotation=90, fontsize=8)

for i in range(11):
    for j in range(4):
        v = data_mat[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                color='white' if v > 0.6 else '#333',
                fontweight='bold' if v > 0.8 else 'normal')

plt.tight_layout()
plt.savefig(OUT / 'fig7_heatmap.png')
plt.close()
print("Figure 7 saved.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 8 — Baseline Comparison: mBERT vs XLM-T vs Indo-RoBERTa
# ══════════════════════════════════════════════════════════════════
bl_exps   = ['MB-E1', 'E1', 'IR-E1']
bl_lbls   = ['mBERT (no DAPT)', 'XLM-T (no DAPT)', 'Indo-RoBERTa (no DAPT)']
bl_colors = [C4, C1, C3]

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)
axes = axes.flatten()
for j, (dskey, ax) in enumerate(zip(DS, axes)):
    ms = [m(e, dskey) for e in bl_exps]
    ss = [s(e, dskey) for e in bl_exps]
    y  = np.arange(len(bl_exps))
    ax.barh(y, ms, color=bl_colors, alpha=0.85, height=0.55, zorder=3)
    ax.errorbar(ms, y, xerr=ss, fmt='none', color='#555', capsize=3, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(bl_lbls)
    ax.invert_yaxis()
    ax.set_title(DS_L[j])
    ax.set_xlabel('Macro-F1')
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig(OUT / 'fig8_baseline_comparison.png')
plt.close()
print("Figure 8 saved.")

print("\nAll figures saved to:", OUT)
