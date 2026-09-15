#!/usr/bin/env python3
"""
chr10_test.py
Directly tests whether the inverse PTEN-glutaminolysis association is driven by
physical co-deletion of signature genes on chromosome 10 near PTEN.

Signature genes by location:
  Chr10 (near PTEN 10q23.31):  GLUD1 (10q23.2), GOT1 (10q24.2)
  Off-chr10:                    GLS (2q32.2), GPT2 (16q11.2), SLC1A5 (19q13.32)

For each included tumor type it computes the Spearman correlation between PTEN
loss severity and each gene, and contrasts a chr10 sub-signature against an
off-chr10 sub-signature. If co-deletion drives the effect, the chr10 genes
should fall much more steeply with PTEN loss than the off-chr10 genes.
Also correlates each signature gene's own copy number with PTEN copy number.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import analyze as A

CHR10 = ["GLUD1", "GOT1"]
OFFCHR10 = ["GLS", "GPT2", "SLC1A5"]


def sub_signature(expr, genes, cols):
    present = [g for g in genes if g in expr.index]
    if not present:
        return None
    sub = np.log2(expr.loc[present, cols].astype(float).clip(lower=0) + 1)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
    return z.mean(axis=0)


def run(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows, gene_rows = [], []
    study_dirs = [os.path.join(datadir, d) for d in sorted(os.listdir(datadir))
                  if os.path.isdir(os.path.join(datadir, d))]
    for sd in study_dirs:
        study = os.path.basename(sd)
        label = A.STUDY_LABELS.get(study, study)
        expr = A.load_expression(sd)
        cna = A.load_cna(sd)
        dosage = A.call_pten_dosage(expr, cna)
        common = dosage.dropna().index.intersection(expr.columns)
        d = dosage[common]
        n = {t: int((d == t).sum()) for t in A.TIERS}
        if not (n["HemDel"] >= A.MIN_TIER and n["HomDel"] >= A.MIN_TIER and n["Intact"] >= A.MIN_INTACT):
            continue
        sev = d.map(A.SEVERITY).astype(float)

        chr10 = sub_signature(expr, CHR10, list(common))
        off = sub_signature(expr, OFFCHR10, list(common))
        if chr10 is None or off is None:
            continue
        r10, p10 = stats.spearmanr(sev, chr10)
        roff, poff = stats.spearmanr(sev, off)
        rows.append({"study": label,
                     "rho_chr10_genes": r10, "p_chr10": p10,
                     "rho_offchr10_genes": roff, "p_offchr10": poff,
                     "difference": r10 - roff})

        # per-gene: expression vs severity, and gene CN vs PTEN CN
        pten_cn = cna.loc["PTEN", common].astype(float) if "PTEN" in cna.index else None
        for g in CHR10 + OFFCHR10:
            if g not in expr.index:
                continue
            gl = np.log2(expr.loc[g, common].astype(float).clip(lower=0) + 1)
            rg, pg = stats.spearmanr(sev, gl)
            cn_r = np.nan
            if pten_cn is not None and g in cna.index:
                gcn = cna.loc[g, common].astype(float)
                if gcn.notna().sum() > 10 and gcn.std() > 0:
                    cn_r = stats.spearmanr(pten_cn, gcn, nan_policy="omit").correlation
            gene_rows.append({"study": label, "gene": g,
                              "chr": "chr10" if g in CHR10 else "other",
                              "rho_expr_vs_severity": rg, "p": pg,
                              "rho_geneCN_vs_ptenCN": cn_r})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "chr10_subsignature_by_study.csv"), index=False)
    gene = pd.DataFrame(gene_rows)
    gene.to_csv(os.path.join(outdir, "chr10_per_gene.csv"), index=False)

    # Figure: chr10 vs off-chr10 rho, paired by study
    if len(res):
        order = res.sort_values("rho_chr10_genes")
        y = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(order) + 1.8), dpi=300)
        for yi, (_, r) in zip(y, order.iterrows()):
            ax.plot([r["rho_offchr10_genes"], r["rho_chr10_genes"]], [yi, yi],
                    color="#bbb", lw=1.5, zorder=1)
        ax.scatter(order["rho_offchr10_genes"], y, c="#E0A458", s=52,
                   edgecolor="black", linewidth=0.5, label="Off-chr10 genes (GLS, GPT2, SLC1A5)", zorder=3)
        ax.scatter(order["rho_chr10_genes"], y, c="#B03A2E", s=52,
                   edgecolor="black", linewidth=0.5, label="Chr10 genes (GLUD1, GOT1)", zorder=3)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(order["study"], fontsize=9)
        ax.set_xlabel("Spearman rho vs PTEN loss severity")
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        ax.margins(y=0.03); plt.tight_layout(); fig.subplots_adjust(left=0.16, right=0.97)
        fig.savefig(os.path.join(outdir, "fig5_chr10_codeletion.png"))
        plt.close(fig)
    return res, gene


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    a = ap.parse_args()
    res, gene = run(a.datadir, a.outdir)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== chr10 vs off-chr10 sub-signature (rho vs PTEN severity) ===")
    print(res.to_string(index=False))
    print("\n=== per-gene: expression vs severity, and gene-CN vs PTEN-CN ===")
    print(gene.to_string(index=False))
    print("\nMean difference (chr10 minus off-chr10) rho:", round(res["difference"].mean(), 3))
