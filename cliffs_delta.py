"""Cliff's delta effect sizes for the PTEN glutaminolysis co-deletion analysis.

For every cohort, computes Cliff's delta with percentile bootstrap 95 percent
confidence intervals for the glutaminolysis signature score and for each
signature gene, comparing PTEN-intact tumors with hemizygously deleted tumors
and with homozygously deleted tumors.

Cliff's delta is the probability that a randomly chosen tumor from the second
group scores lower than one from the first, minus the probability of the
reverse. Negative values mean the deleted tier scores lower.

Input:  minimal_dataset/minimal_dataset_tumor_level.csv
Output: results/cliffs_delta_signature.csv
        results/cliffs_delta_by_gene.csv

Usage:  python cliffs_delta.py
"""

import os
import numpy as np
import pandas as pd

DATA = os.path.join("minimal_dataset", "minimal_dataset_tumor_level.csv")
OUTDIR = "results"
GENES = ["GLUD1", "GOT1", "GLS", "GPT2", "SLC1A5"]
N_BOOT = 2000
SEED = 20260917


def cliffs_delta(a, b):
    """Cliff's delta of b relative to a, computed by rank identity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return np.nan
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().values
    rb = ranks[len(a):].sum()
    u = rb - len(b) * (len(b) + 1) / 2.0
    return 2.0 * u / (len(a) * len(b)) - 1.0


def boot_ci(a, b, rng, n=N_BOOT):
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    out = np.empty(n)
    for i in range(n):
        out[i] = cliffs_delta(rng.choice(a, len(a), replace=True),
                             rng.choice(b, len(b), replace=True))
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


def rows_for(df, value_col, label, rng):
    rows = []
    cohorts = sorted(df["study"].unique())
    for i, study in enumerate(cohorts, start=1):
        sub = df[df["study"] == study]
        intact = sub.loc[sub.pten_tier == "Intact", value_col].dropna().values
        print("  [%2d/%2d] %s %s (intact n = %d)"
              % (i, len(cohorts), study, label, len(intact)), flush=True)
        for tier, name in (("HemDel", "hemizygous"), ("HomDel", "homozygous")):
            other = sub.loc[sub.pten_tier == tier, value_col].dropna().values
            d = cliffs_delta(intact, other)
            lo, hi = boot_ci(intact, other, rng)
            rows.append(dict(cancer=study, measure=label, comparison="intact vs " + name,
                             n_intact=len(intact), n_other=len(other),
                             cliffs_delta=d, ci_low=lo, ci_high=hi))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA)
    os.makedirs(OUTDIR, exist_ok=True)

    print("signature score")
    sig = pd.DataFrame(rows_for(df, "signature_score", "signature", rng))
    sig.round(4).to_csv(os.path.join(OUTDIR, "cliffs_delta_signature.csv"), index=False)

    allrows = []
    for gene in GENES:
        print("gene %s" % gene)
        allrows += rows_for(df, "z_" + gene, gene, rng)
    pd.DataFrame(allrows).round(4).to_csv(
        os.path.join(OUTDIR, "cliffs_delta_by_gene.csv"), index=False)

    print("\nwrote both result files")
    hom = sig[sig.comparison == "intact vs homozygous"]
    print("signature, intact vs homozygous: median delta %.3f, range %.3f to %.3f"
          % (hom.cliffs_delta.median(), hom.cliffs_delta.min(), hom.cliffs_delta.max()))


if __name__ == "__main__":
    main()
