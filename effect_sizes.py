#!/usr/bin/env python3
"""
effect_sizes.py

Closes Reviewer 2, observation 4.

The submitted manuscript reports Spearman correlations and adjusted P values
and describes the result as a uniform pattern across the included tumor types. Two of
those correlations are -0.06 and -0.07, which explain well under one percent of
variance, and the dosage boxplots show visibly non-monotonic behaviour in breast and
glioblastoma. A P value below 0.05 on a cohort of 1068 tumors says the effect is
not zero; it says nothing about whether the effect matters.

This module reports, for every gene and every cohort:

  * the median difference between intact and deleted tumors, with a bootstrap
    95 percent confidence interval
  * Cliff's delta, a rank-based effect size bounded at plus or minus one, with
    its own bootstrap interval
  * the variance explained by the correlation, so trivial associations are
    visible as trivial
  * a monotonicity check, since a correlation summarises a trend that may not
    actually be monotonic across the three tiers

The monotonicity column is the one to read alongside Figure 2. A negative
Spearman coefficient is compatible with the middle tier sitting above the
intact tier, which is what the reviewer objected to.

Usage:
    python effect_sizes.py --datadir ./data --outdir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A

N_BOOT = 2000
RNG = np.random.default_rng(20260912)
TRIVIAL_RHO = 0.10


def cliffs_delta(a, b):
    """Rank-based effect size. Positive means a tends to exceed b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    n1, n2 = a.size, b.size
    ranks = stats.rankdata(np.concatenate([a, b]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(2 * u1 / (n1 * n2) - 1)


def _boot_two_sample(fn, a, b, n_boot=N_BOOT, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = fn(RNG.choice(a, a.size, replace=True),
                     RNG.choice(b, b.size, replace=True))
    vals = vals[np.isfinite(vals)]
    if vals.size < n_boot // 4:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def _monotonic(med_intact, med_hem, med_hom):
    """Describe the three-tier pattern rather than assuming it is monotonic."""
    vals = [med_intact, med_hem, med_hom]
    if any(not np.isfinite(v) for v in vals):
        return "incomplete"
    if med_intact >= med_hem >= med_hom:
        return "monotonic decreasing"
    if med_intact <= med_hem <= med_hom:
        return "monotonic increasing"
    return "non-monotonic"


def run(datadir, outdir, genes=None):
    genes = genes or A.GLUTAMINOLYSIS_GENES
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for d in sorted(os.listdir(datadir)):
        sdir = os.path.join(datadir, d)
        if not os.path.isdir(sdir):
            continue
        label = A.STUDY_LABELS.get(d, d)
        try:
            expr = A.load_expression(sdir)
            cna = A.load_cna(sdir)
        except Exception as exc:
            print(f"  [skip] {label}: {exc}")
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
        sev = tiers.map(A.SEVERITY).astype(float).values

        present = [g for g in genes if g in expr.index]
        sub = np.log2(expr.loc[present, cols].astype(float).clip(lower=0) + 1)
        z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)

        targets = {g: z.loc[g].values for g in present}
        targets["signature"] = z.mean(axis=0).values

        for name, vals in targets.items():
            intact = vals[tiers.values == "Intact"]
            hem = vals[tiers.values == "HemDel"]
            hom = vals[tiers.values == "HomDel"]

            rho, p = stats.spearmanr(sev, vals, nan_policy="omit")
            med_diff = float(np.nanmedian(hom) - np.nanmedian(intact))
            lo, hi = _boot_two_sample(
                lambda x, y: float(np.median(x) - np.median(y)), hom, intact)
            delta = cliffs_delta(hom, intact)
            d_lo, d_hi = _boot_two_sample(cliffs_delta, hom, intact)

            rows.append({
                "study": label,
                "target": name,
                "n_intact": int(intact.size),
                "n_hem": int(hem.size),
                "n_hom": int(hom.size),
                "median_intact": round(float(np.nanmedian(intact)), 4),
                "median_hem": round(float(np.nanmedian(hem)), 4),
                "median_hom": round(float(np.nanmedian(hom)), 4),
                "pattern": _monotonic(np.nanmedian(intact), np.nanmedian(hem),
                                      np.nanmedian(hom)),
                "median_diff_hom_vs_intact": round(med_diff, 4),
                "median_diff_ci_lo": None if np.isnan(lo) else round(lo, 4),
                "median_diff_ci_hi": None if np.isnan(hi) else round(hi, 4),
                "cliffs_delta_hom_vs_intact": None if np.isnan(delta) else round(delta, 4),
                "cliffs_delta_ci_lo": None if np.isnan(d_lo) else round(d_lo, 4),
                "cliffs_delta_ci_hi": None if np.isnan(d_hi) else round(d_hi, 4),
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p),
                "variance_explained_pct": round(100 * float(rho) ** 2, 2),
                "trivial": abs(float(rho)) < TRIVIAL_RHO,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[effect_sizes] no cohort met the inclusion rule")
        return res

    out = os.path.join(outdir, "effect_sizes.csv")
    res.to_csv(out, index=False)
    print(f"[effect_sizes] {len(res)} rows -> {out}")

    sig = res[res["target"] == "signature"].sort_values("spearman_rho")
    print("\nSignature, by cohort:")
    print(sig[["study", "n_intact", "n_hom", "spearman_rho",
               "variance_explained_pct", "cliffs_delta_hom_vs_intact",
               "pattern"]].to_string(index=False))

    trivial = sig[sig["trivial"]]
    if len(trivial):
        print(f"\n  Correlations below |rho| = {TRIVIAL_RHO} in: "
              f"{', '.join(trivial['study'])}")
        print("  These explain under one percent of variance and should not be "
              "described as part of a uniform pattern.")
    nonmono = sig[sig["pattern"] == "non-monotonic"]
    if len(nonmono):
        print(f"  Non-monotonic across the three tiers: "
              f"{', '.join(nonmono['study'])}")
        print("  A negative Spearman coefficient does not imply monotonicity; "
              "these cohorts need describing as they are.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    args = ap.parse_args()
    run(args.datadir, args.outdir)


if __name__ == "__main__":
    main()
