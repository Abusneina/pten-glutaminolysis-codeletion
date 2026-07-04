#!/usr/bin/env python3
"""
depmap_module.py  --  Aim 2 (three-tier PTEN dosage).
Test whether GLS (glutaminase) genetic dependency scales with PTEN
copy-number dosage across cancer cell lines, using DepMap Public data.

DepMap is not on the GitHub allowlist, so obtain these files from
https://depmap.org/portal/download/ (free account) and pass their paths:
  * CRISPRGeneEffect.csv   (rows: ModelID; cols: 'GENE (Entrez)'); Chronos scores
  * OmicsCNGene.csv        (gene-level copy number)  [required for dosage]
  * Model.csv              (lineage metadata)         [optional]

Copy-number units differ between DepMap releases. OmicsCNGene is typically
RELATIVE copy number where ~1.0 is neutral (two copies). Defaults below
assume that convention; verify against your release and adjust --hom_max
and --hem_max if needed. A more negative Chronos score = stronger dependency.

Usage:
  python3 depmap_module.py --gene_effect CRISPRGeneEffect.csv \
      --cn OmicsCNGene.csv --model Model.csv --outdir results
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TIERS = ["Intact", "HemDel", "HomDel"]
SEVERITY = {"Intact": 0, "HemDel": 1, "HomDel": 2}
TIER_COLORS = {"Intact": "#2E9599", "HemDel": "#E0A458", "HomDel": "#B03A2E"}


def _find_gene_col(df, gene):
    for c in df.columns:
        if re.match(rf"^{gene}\b", str(c)):
            return c
    return None


def classify_cn(value, hom_max, hem_max):
    """Relative CN -> tier. <hom_max: HomDel; <hem_max: HemDel; else Intact."""
    if pd.isna(value):
        return np.nan
    if value < hom_max:
        return "HomDel"
    if value < hem_max:
        return "HemDel"
    return "Intact"


def run(gene_effect, cn, model, outdir, gene="GLS", hom_max=0.25, hem_max=0.75):
    os.makedirs(outdir, exist_ok=True)
    ge = pd.read_csv(gene_effect, index_col=0)
    col = _find_gene_col(ge, gene)
    if col is None:
        raise SystemExit(f"{gene} not found in gene-effect matrix")
    dep = ge[col].rename("gene_effect")

    if not (cn and os.path.exists(cn)):
        raise SystemExit("OmicsCNGene.csv is required for dosage classification")
    cndf = pd.read_csv(cn, index_col=0)
    pcol = _find_gene_col(cndf, "PTEN")
    if pcol is None:
        raise SystemExit("PTEN not found in copy-number matrix")
    tier = cndf[pcol].apply(lambda v: classify_cn(v, hom_max, hem_max)).rename("tier")

    df = dep.to_frame().join(tier, how="inner").dropna(subset=["gene_effect", "tier"])
    if model and os.path.exists(model):
        md = pd.read_csv(model, index_col=0)
        lin = next((c for c in md.columns if c in
                    ("OncotreeLineage", "lineage", "OncotreePrimaryDisease")), None)
        if lin:
            df["lineage"] = md[lin].reindex(df.index)

    groups = {t: df[df.tier == t]["gene_effect"] for t in TIERS}
    n = {t: len(groups[t]) for t in TIERS}
    present = [t for t in TIERS if n[t] >= 3]
    kw_stat, kw_p = stats.kruskal(*[groups[t] for t in present])
    sev = df["tier"].map(SEVERITY).astype(float)
    rho, rho_p = stats.spearmanr(sev, df["gene_effect"])  # expect negative rho

    res = pd.DataFrame([{
        "gene": gene,
        "n_intact": n["Intact"], "n_hemdel": n["HemDel"], "n_homdel": n["HomDel"],
        "median_effect_intact": groups["Intact"].median() if n["Intact"] else np.nan,
        "median_effect_hemdel": groups["HemDel"].median() if n["HemDel"] else np.nan,
        "median_effect_homdel": groups["HomDel"].median() if n["HomDel"] else np.nan,
        "kruskal_H": kw_stat, "kruskal_p": kw_p,
        "spearman_rho_severity": rho, "spearman_p": rho_p,
    }])
    res.to_csv(os.path.join(outdir, "aim2_depmap_dose_response.csv"), index=False)
    df.to_csv(os.path.join(outdir, "aim2_depmap_perline.csv"))

    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=300)
    for i, tier in enumerate(TIERS):
        vals = groups[tier].dropna()
        if len(vals) == 0:
            continue
        bp = ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=TIER_COLORS[tier], alpha=0.8, edgecolor="black")
        for m in bp["medians"]:
            m.set(color="black")
        ax.scatter(np.random.normal(i, 0.05, len(vals)), vals, s=6, color="black", alpha=0.3)
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([f"{t}\n(n={n[t]})" for t in TIERS])
    ax.set_ylabel(f"{gene} CRISPR gene effect (Chronos)")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title(f"{gene} dependency across PTEN copy-number dosage (DepMap)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_depmap_gls_dose_response.png"))
    plt.close(fig)
    print(res.to_string(index=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_effect", required=True)
    ap.add_argument("--cn", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gene", default="GLS")
    ap.add_argument("--hom_max", type=float, default=0.25,
                    help="relative CN below this = HomDel")
    ap.add_argument("--hem_max", type=float, default=0.75,
                    help="relative CN below this (and >= hom_max) = HemDel")
    ap.add_argument("--outdir", default="./results")
    a = ap.parse_args()
    run(a.gene_effect, a.cn, a.model, a.outdir, a.gene, a.hom_max, a.hem_max)


if __name__ == "__main__":
    main()
