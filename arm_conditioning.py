"""10q arm-level conditioning for the PTEN glutaminolysis co-deletion analysis.

Recomputes the association between PTEN loss severity and each signature gene's
expression while conditioning on a 10q arm-level proxy, and compares it with the
unadjusted association and with conditioning on each gene's own copy number.

The arm-level proxy is the median thresholded GISTIC copy number of the 10q23-q24
interval genes lying more than 2 Mb from PTEN, so that the focal events which
remove PTEN and its immediate neighbours do not themselves define the covariate.

Input:  minimal_dataset/minimal_dataset_tumor_level.csv
Output: results/arm_conditioning.csv  (per cohort and gene)
        summary printed to stdout

Usage:  python arm_conditioning.py
"""

import os
import numpy as np
import pandas as pd

DATA = os.path.join("minimal_dataset", "minimal_dataset_tumor_level.csv")
OUTDIR = "results"
OUTFILE = os.path.join(OUTDIR, "arm_conditioning.csv")

SIGNATURE = ["GLUD1", "GOT1", "GLS", "GPT2", "SLC1A5"]
CHR10 = ["GLUD1", "GOT1"]
# interval genes more than 2 Mb from the PTEN transcription start site
ARM_PROXY = ["cn_LIPA", "cn_HHEX", "cn_IDE", "cn_KIF11"]


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z, via rank inverse covariance."""
    m = pd.concat([x, y, z], axis=1).dropna()
    if len(m) < 20 or m.iloc[:, 2].nunique() < 2:
        return np.nan, len(m)
    r = m.rank()
    c = np.corrcoef(r.T.values)
    try:
        p = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        return np.nan, len(m)
    return -p[0, 1] / np.sqrt(p[0, 0] * p[1, 1]), len(m)


def spearman(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 20:
        return np.nan, len(m)
    r = m.rank()
    return np.corrcoef(r.T.values)[0, 1], len(m)


def main():
    df = pd.read_csv(DATA)
    missing = [c for c in ARM_PROXY if c not in df.columns]
    if missing:
        raise SystemExit("missing arm proxy columns: %s" % missing)
    df["arm10q"] = df[ARM_PROXY].median(axis=1)

    rows = []
    cohorts = sorted(df["study"].unique())
    for i, study in enumerate(cohorts, start=1):
        sub = df[df["study"] == study]
        print("[%2d/%2d] %s (n = %d)" % (i, len(cohorts), study, len(sub)), flush=True)
        for gene in SIGNATURE:
            unadj, n = spearman(sub["z_" + gene], sub["pten_severity"])
            arm, _ = partial_spearman(sub["z_" + gene], sub["pten_severity"], sub["arm10q"])
            own, _ = partial_spearman(sub["z_" + gene], sub["pten_severity"], sub["cn_" + gene])
            rows.append(dict(cancer=study, gene=gene, n=n,
                             chromosome="chr10" if gene in CHR10 else "other",
                             rho_unadjusted=unadj, rho_given_arm10q=arm,
                             rho_given_own_cn=own))

    out = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    out.round(4).to_csv(OUTFILE, index=False)
    print("\nwrote %s" % OUTFILE)

    for label, genes in (("chromosome 10", CHR10),
                         ("off chromosome 10", [g for g in SIGNATURE if g not in CHR10])):
        s = out[out.gene.isin(genes)]
        print("%-18s unadjusted %+.3f | given 10q arm %+.3f | given own CN %+.3f"
              % (label, s.rho_unadjusted.mean(), s.rho_given_arm10q.mean(),
                 s.rho_given_own_cn.mean()))


if __name__ == "__main__":
    main()
