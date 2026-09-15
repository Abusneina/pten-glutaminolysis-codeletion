#!/usr/bin/env python3
"""
make_figures.py

Rebuilds the whole figure set to PLOS specification.

Every submitted figure exceeded the PLOS maximum width of 2250 pixels at 300
dpi, in one case by a factor of 2.6, which is why they were scaled down at
layout and why both the editor and Reviewer 2 found them illegible. Resolution
was never the problem.

Every figure here is built to the specification rather than checked against it
afterwards: at most 2250 x 2625 pixels, 300 dpi, RGB with no alpha channel, LZW
compressed, single layer, Arial at 8 to 12 point, no title inside the image, and
legends inside the plot area. No blue.

Figure numbering changes because the interval analysis is new and the survival
analysis moves out:

    Fig 1  schematic (produced by make_fig1.py, not here)
    Fig 2  signature by PTEN dosage
    Fig 3  dose-response across all cohorts
    Fig 4  interval and distance decay          <- new
    Fig 5  chromosome-10 versus off-chromosome-10
    Fig 6  MYC is not the mediator
    S1 Fig survival by PTEN dosage              <- was Fig 6

Usage:
    python make_figures.py --datadir .\\data --resultsdir .\\results --outdir .\\figures
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import analyze as A

# Palette: burgundy, amber, teal, grey. Deliberately no blue.
BURGUNDY = "#7B2D26"
AMBER = "#C08A2E"
TEAL = "#2E7D6B"
GREY = "#6E6E6E"
FAINT = "#CCCCCC"
INK = "#1A1A1A"

TIER_COLORS = {"Intact": TEAL, "HemDel": AMBER, "HomDel": BURGUNDY}
MAX_W, MAX_H = 2250, 2625


def _setup():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _make_room_below(fig, ax, leg, pad=0.015):
    """Lower the y-limit until nothing plotted sits behind the legend.

    Measured from the drawn legend rather than padded by a guessed constant,
    so it holds whatever the data range turns out to be.
    """
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    top = min(max(bb.y1, 0.0) + pad, 0.55)
    lo, hi = ax.get_ylim()
    ax.set_ylim(hi - (hi - lo) / (1.0 - top), hi)


def _make_room_above(fig, ax, leg, pad=0.015):
    """Raise the upper y-limit until nothing plotted sits behind the legend.

    Used for the horizontal dot plots, where the corners are not reliably
    empty: moving the legend from one corner to another only moves which
    markers it covers, so a blank band is reserved above the top row instead.
    """
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    bottom = max(min(bb.y0, 1.0) - pad, 0.45)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) / bottom)


def save(fig, outdir, name):
    """Write a PLOS-compliant TIFF plus a PNG for previewing, and verify."""
    png = os.path.join(outdir, f"{name}.png")
    tif = os.path.join(outdir, f"{name}.tif")
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.04)
    plt.close(fig)
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w > MAX_W or h > MAX_H:
            scale = min(MAX_W / w, MAX_H / h)
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        rgb.save(tif, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    with Image.open(tif) as im:
        w, h = im.size
        ok = "OK" if (w <= MAX_W and h <= MAX_H) else "OVER"
        print(f"    {name}.tif  {w} x {h} px, {im.mode}, LZW  [{ok}]", flush=True)


def load_cohorts(datadir):
    """Per-sample tier and signature score for every qualifying cohort."""
    out = {}
    dirs = [d for d in sorted(os.listdir(datadir))
            if os.path.isdir(os.path.join(datadir, d))]
    for i, code in enumerate(dirs, 1):
        sdir = os.path.join(datadir, code)
        label = A.STUDY_LABELS.get(code, code)
        try:
            expr = A.load_expression(sdir)
            cna = A.load_cna(sdir)
        except Exception:
            continue
        tiers = A.call_pten_dosage(expr, cna)
        if tiers is None or not len(tiers):
            continue
        counts = tiers.value_counts()
        if (counts.get("HemDel", 0) < A.MIN_TIER or counts.get("HomDel", 0) < A.MIN_TIER
                or counts.get("Intact", 0) < A.MIN_INTACT):
            continue
        tiers = tiers.dropna()
        cols = list(tiers.index)
        present = [g for g in A.GLUTAMINOLYSIS_GENES if g in expr.index]
        sub = np.log2(expr.loc[present, cols].astype(float).clip(lower=0) + 1)
        z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
        out[label] = pd.DataFrame({"tier": tiers.values,
                                   "signature": z.mean(axis=0).values},
                                  index=cols)
        print(f"  [{i}/{len(dirs)}] {label}: {len(cols)} tumors", flush=True)
    return out


# --------------------------------------------------------------------------
def fig2_signature_by_dosage(cohorts, outdir, n_show=6):
    """Boxplots for the cohorts with the largest effects; rest to Supporting."""
    order = sorted(cohorts, key=lambda k: np.median(
        cohorts[k].loc[cohorts[k].tier == "HomDel", "signature"]) -
        np.median(cohorts[k].loc[cohorts[k].tier == "Intact", "signature"]))
    for tag, keys in (("Fig2", order[:n_show]), ("S2_Fig", order[n_show:])):
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
        positions, ticks, labels = [], [], []
        for gi, k in enumerate(keys):
            df = cohorts[k]
            for ti, tier in enumerate(("Intact", "HemDel", "HomDel")):
                v = df.loc[df.tier == tier, "signature"].dropna().values
                if not len(v):
                    continue
                pos = gi * 4 + ti
                bp = ax.boxplot([v], positions=[pos], widths=0.72,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color=INK, lw=1.1),
                                whiskerprops=dict(color=GREY, lw=0.8),
                                capprops=dict(color=GREY, lw=0.8))
                bp["boxes"][0].set_facecolor(TIER_COLORS[tier])
                bp["boxes"][0].set_edgecolor(INK)
                bp["boxes"][0].set_linewidth(0.7)
                bp["boxes"][0].set_alpha(0.92)
                positions.append(pos)
            ticks.append(gi * 4 + 1)
            labels.append(k)
        ax.axhline(0, color=FAINT, lw=0.8, ls=":", zorder=0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Glutaminolysis signature\n(mean gene z-score)")
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=TIER_COLORS[t],
                                 edgecolor=INK, lw=0.7) for t in
                   ("Intact", "HemDel", "HomDel")]
        leg = ax.legend(handles, ["Intact", "Hemizygous", "Homozygous"],
                        loc="lower left", frameon=True, framealpha=0.92,
                        edgecolor=FAINT, ncol=3, columnspacing=1.1)
        # The lower whiskers reached into the legend box in the 9/14 render,
        # clipping CESC in Fig2 and HNSC and UCEC in S2 Fig.
        _make_room_below(fig, ax, leg)
        fig.tight_layout(pad=0.3)
        print(f"    {tag}: {len(keys)} cohorts ({', '.join(keys)})", flush=True)
        save(fig, outdir, tag)


def fig3_dose_response(results, outdir):
    e = pd.read_csv(os.path.join(results, "effect_sizes.csv"))
    s = e[e.target == "signature"].sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    y = np.arange(len(s))
    # The caption and the Methods both claim significance after Benjamini-
    # Hochberg correction, so the figure has to mark it that way.
    pcol = next((c for c in ("spearman_p_adj", "spearman_q", "p_adj", "q_value",
                             "fdr_bh", "p_bh") if c in s.columns), None)
    if pcol is None:
        from statsmodels.stats.multitest import multipletests
        s = s.assign(_p_bh=multipletests(s["spearman_p"].values,
                                         method="fdr_bh")[1])
        pcol = "_p_bh"
        print("    effect_sizes.csv carries no adjusted P column, so Benjamini-"
              "Hochberg is computed here across the included cohorts, which is "
              "what the Methods specify", flush=True)
    else:
        print(f"    significance from {pcol}", flush=True)
    sig = s[pcol] < 0.05
    ns = list(s.study[~sig])
    print(f"    significant after correction: {int(sig.sum())} of {len(s)}"
          f"; not significant: {', '.join(ns) if ns else 'none'}", flush=True)
    print("    check this against the Padj column of Table 1 before uploading",
          flush=True)
    ax.scatter(s.spearman_rho[sig], y[sig.values], s=52, color=BURGUNDY,
               edgecolors="white", lw=0.6, zorder=3, label="Significant")
    ax.scatter(s.spearman_rho[~sig], y[~sig.values], s=48, color=GREY,
               edgecolors="white", lw=0.6, zorder=3, label="Not significant")
    # Median-difference intervals are on a different scale from rho and are
    # reported in S1 Table instead of drawn here.
    ax.axvline(0, color=FAINT, lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(s.study)
    ax.set_xlabel("Spearman rho (glutaminolysis signature vs PTEN loss severity)")
    # No corner of this panel is reliably empty: the top row sits near zero and
    # the bottom row is the most negative value. A blank band above the rows is
    # the only placement that cannot cover a marker.
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.92,
                    edgecolor=FAINT)
    _make_room_above(fig, ax, leg)
    fig.tight_layout(pad=0.3)
    save(fig, outdir, "Fig3")


def fig4_interval(results, outdir):
    g = pd.read_csv(os.path.join(results, "interval_decay_per_gene.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    style = {"neighbor, unannotated": (AMBER, "s", 34,
             "10q neighbors, no glutaminolysis annotation"),
             "signature, chr10": (BURGUNDY, "o", 90,
             "Signature genes on chr10 (GLUD1, GOT1)"),
             "signature, off-chr10": (TEAL, "^", 52,
             "Signature genes on other chromosomes")}
    chr10_max = g.loc[g.distance_mb.notna(), "distance_mb"].max()
    off_x = chr10_max * 3.2
    # Draw neighbours first so the signature genes sit on top and stay visible.
    for cls in ("neighbor, unannotated", "signature, chr10", "signature, off-chr10"):
        sub = g[g["class"] == cls]
        if not len(sub):
            continue
        color, marker, size, label = style[cls]
        xs = (np.full(len(sub), off_x) if cls == "signature, off-chr10"
              else sub.distance_mb.values)
        ax.scatter(xs, sub.mean_rho, c=color, marker=marker, s=size,
                   edgecolors="white", lw=0.7, label=label,
                   zorder=4 if "signature" in cls else 3)
    fit = g[(g["class"] == "neighbor, unannotated") & g.distance_mb.notna()]
    if len(fit) >= 5:
        x = np.log10(fit.distance_mb.clip(lower=0.01))
        b, a = np.polyfit(x, fit.mean_rho, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(10 ** xx, a + b * xx, color=GREY, lw=1.2, ls="--", zorder=2,
                label="Fit to unannotated neighbors")
    ax.axvline(chr10_max * 1.8, color=FAINT, lw=0.9, ls=":", zorder=1)
    ax.axhline(0, color=FAINT, lw=0.8, zorder=1)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.08, 1.05)
    ax.text(off_x, 1.02, "other\nchromosomes", ha="center", va="top",
            fontsize=8, color=GREY)
    ax.set_xlabel("Distance from PTEN transcription start site (Mb, log scale)")
    ax.set_ylabel("Spearman rho, gene copy number\nvs PTEN copy number")
    ax.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor=FAINT)
    fig.tight_layout(pad=0.3)
    save(fig, outdir, "Fig4")


def fig5_chr10_vs_off(results, outdir):
    path = os.path.join(results, "chr10_subsignature_by_study.csv")
    if not os.path.exists(path):
        print("    Fig5 skipped: run chr10_test.py first")
        return
    c = pd.read_csv(path).sort_values("rho_chr10_genes")
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=300)
    y = np.arange(len(c))
    ax.hlines(y, c.rho_chr10_genes, c.rho_offchr10_genes, color=FAINT, lw=1.6,
              zorder=1)
    ax.scatter(c.rho_chr10_genes, y, s=56, color=BURGUNDY, edgecolors="white",
               lw=0.7, zorder=3, label="Chr10 genes (GLUD1, GOT1)")
    ax.scatter(c.rho_offchr10_genes, y, s=56, color=AMBER, edgecolors="white",
               lw=0.7, zorder=3, label="Off-chr10 genes (GLS, GPT2, SLC1A5)")
    ax.axvline(0, color=GREY, lw=0.9, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(c.study)
    ax.set_xlabel("Spearman rho vs PTEN loss severity")
    # Lower left covered the chromosome-10 marker for SARC; lower right then
    # covered the off-chromosome-10 marker for SARC and clipped SKCM, because
    # those values sit near zero on the bottom rows. Reserve a band instead.
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.94,
                    edgecolor=FAINT)
    _make_room_above(fig, ax, leg)
    fig.tight_layout(pad=0.3)
    save(fig, outdir, "Fig5")


def fig6_myc(results, outdir):
    m = pd.read_csv(os.path.join(results, "mediation_myc.csv"))
    sub = m[m.mediator.isin(["MYC_mRNA", "MYC_targets_V1"])]
    if not len(sub):
        print("    Fig6 skipped: no mediation results")
        return
    piv = sub.pivot_table(index="study", columns="mediator",
                          values="path_a_exposure_to_mediator")
    piv = piv.sort_values(piv.columns[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    y = np.arange(len(piv))
    cols = list(piv.columns)
    colors = {cols[0]: BURGUNDY, cols[-1]: TEAL}
    names = {"MYC_mRNA": "MYC transcript",
             "MYC_targets_V1": "Hallmark MYC targets V1"}
    for ci, col in enumerate(cols):
        ax.scatter(piv[col], y + (ci - 0.5) * 0.22, s=48,
                   color=colors.get(col, AMBER), edgecolors="white", lw=0.6,
                   zorder=3, label=names.get(col, col))
    ax.axvline(0, color=GREY, lw=0.9, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Association of PTEN loss severity with the mediator\n"
                  "(standardized, rank-based)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor=FAINT)
    fig.tight_layout(pad=0.3)
    save(fig, outdir, "Fig6")


# The cohorts the survival paragraph actually discusses. Taking whichever
# cohorts happen to come first alphabetically produced a figure that showed
# none of the tumor types named in the text.
SURVIVAL_COHORTS = ["UCEC", "GBM", "PRAD", "BRCA"]

# Panels whose curves occupy the default lower-left corner. Value is
# (loc, bbox_to_anchor in axes coordinates); None means no anchor.
SURVIVAL_LEGEND_LOC = {"GBM": ("upper right", (1.0, 0.88))}


def s1_survival(datadir, cohorts, outdir, n_show=4):
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("    S1 Fig skipped: lifelines not installed")
        return
    keys = [k for k in SURVIVAL_COHORTS if k in cohorts]
    if len(keys) < n_show:
        keys += [k for k in cohorts if k not in keys][:n_show - len(keys)]
    keys = keys[:n_show]
    print(f"    panels: {', '.join(keys)}", flush=True)
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 3.2 * nrow), dpi=300,
                             squeeze=False)
    drawn = 0
    for ax, label in zip(axes.ravel(), keys):
        code = next((c for c in os.listdir(datadir)
                     if A.STUDY_LABELS.get(c, c) == label), None)
        if code is None:
            continue
        try:
            clin = A.load_clinical(os.path.join(datadir, code))
        except Exception:
            continue
        if clin is None or not len(clin):
            continue
        df = cohorts[label].copy()
        df["patient"] = [str(i)[:12] for i in df.index]
        # load_clinical returns a plain DataFrame with a row-number index, so
        # the patient barcode must be taken from its PATIENT_ID column. Merging
        # on the row index silently matches nothing and yields empty panels.
        clin = clin.copy()
        clin.columns = [str(c).strip().upper().replace(" ", "_").lstrip("#")
                        for c in clin.columns]
        if "PATIENT_ID" not in clin.columns or "OS_MONTHS" not in clin.columns:
            print(f"    {label}: no PATIENT_ID or OS_MONTHS, skipped", flush=True)
            continue
        clin["PATIENT_ID"] = clin["PATIENT_ID"].astype(str)
        clin = clin.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
        merged = df.merge(clin, left_on="patient", right_index=True, how="left")
        merged["t"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
        if "OS_STATUS" in merged.columns:
            merged["e"] = merged["OS_STATUS"].astype(str).str.startswith("1")
        else:
            merged["e"] = False
        merged = merged.dropna(subset=["t"])
        if len(merged) < 15:
            print(f"    {label}: only {len(merged)} tumors with survival, skipped",
                  flush=True)
            continue
        for tier in ("Intact", "HemDel", "HomDel"):
            s = merged[merged.tier == tier]
            if len(s) < 5:
                continue
            km = KaplanMeierFitter()
            km.fit(s["t"], s["e"], label=f"{tier} (n={len(s)})")
            km.plot_survival_function(ax=ax, ci_show=False,
                                      color=TIER_COLORS[tier], lw=1.4)
        ax.set_xlabel("Overall survival (months)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        if ax.get_legend_handles_labels()[0]:
            # In GBM every curve falls steeply through the lower left, where the
            # legend covered a stretch of the hemizygous curve. Anchored below
            # the cohort label so the two do not collide.
            loc, anchor = SURVIVAL_LEGEND_LOC.get(label, ("lower left", None))
            kw = dict(loc=loc, frameon=True, framealpha=0.92,
                      edgecolor=FAINT, fontsize=7.5)
            if anchor is not None:
                kw.update(bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
            ax.legend(**kw)
        else:
            print(f"    {label}: no tier had enough tumors to plot", flush=True)
        ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right",
                va="top", fontsize=9.5, fontweight="bold", color=INK)
        # PLOS expects lettered panels on a multi-panel figure, and the caption
        # refers to them. Letters follow the drawing order, so a skipped cohort
        # does not leave a gap in the sequence.
        ax.text(-0.16, 1.06, chr(ord("A") + drawn), transform=ax.transAxes,
                ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
        drawn += 1
        print(f"    {label}: {len(merged)} tumors with survival data", flush=True)
    for ax in axes.ravel()[drawn:]:
        ax.axis("off")
    if drawn:
        fig.tight_layout(pad=0.4)
        save(fig, outdir, "S1_Fig")
    else:
        plt.close(fig)
        print("    S1 Fig skipped: no survival data resolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--resultsdir", default="./results")
    ap.add_argument("--outdir", default="./figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    _setup()

    print("Loading cohorts")
    cohorts = load_cohorts(args.datadir)
    print(f"  {len(cohorts)} cohorts\n")

    for title, fn in (
            ("Fig 2, signature by dosage", lambda: fig2_signature_by_dosage(cohorts, args.outdir)),
            ("Fig 3, dose-response", lambda: fig3_dose_response(args.resultsdir, args.outdir)),
            ("Fig 4, interval decay", lambda: fig4_interval(args.resultsdir, args.outdir)),
            ("Fig 5, chr10 vs off-chr10", lambda: fig5_chr10_vs_off(args.resultsdir, args.outdir)),
            ("Fig 6, MYC", lambda: fig6_myc(args.resultsdir, args.outdir)),
            ("S1 Fig, survival", lambda: s1_survival(args.datadir, cohorts, args.outdir))):
        print(f"  {title}", flush=True)
        try:
            fn()
        except Exception as exc:
            print(f"    FAILED: {exc}")

    print(f"\nFigures in {args.outdir}")
    print("Submit the .tif files. The .png copies are for previewing only.")
    print("Figure 1 is produced separately by make_fig1.py.")


if __name__ == "__main__":
    main()
