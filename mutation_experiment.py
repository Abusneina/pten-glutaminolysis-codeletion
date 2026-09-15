#!/usr/bin/env python3
"""
mutation_experiment.py

Reconstructs the mutation-based natural experiment reported as S4 Table:
tumors in which PTEN is inactivated by point mutation or small indel while
retaining neutral or gained copy number are compared against copy-neutral
PTEN-wild-type tumors, and against hemizygously and homozygously deleted
tumors as positive controls.

This analysis is described in the Methods and Results of PONE-D-26-35226 but
was not present in the v1.0 code deposit, although fetch_data.py downloads the
mutation file. The numbers it produces must be checked against the published
S4 Table before the revision is submitted. In particular the pooled P = 0.43
and the endometrial P = 1e-5 need to reproduce.

Beyond reconstruction, three things are added that reviewers asked for:

  * Equivalence testing. A non-significant difference is not evidence of no
    difference. Two one-sided tests against a pre-stated margin are reported
    alongside the Mann-Whitney result, so "unchanged" becomes a claim that can
    actually fail.
  * Per-cohort group sizes for every contrast, not only for endometrial
    carcinoma.
  * Effect sizes with bootstrap confidence intervals, so a trivial difference
    is visible as trivial.

The equivalence margin is a scientific choice and must be fixed before the
result is looked at. The default of 0.25 within-cohort standard deviations is
roughly a quarter of the effect that hemizygous deletion produces, and should
be stated in the Methods as pre-specified.

Usage:
    python3 mutation_experiment.py --datadir ./data --outdir ./results
    python3 mutation_experiment.py --datadir ./data --outdir ./results --truncating-only
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A

GENE_OF_INTEREST = "GLUD1"

INACTIVATING = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"}

DEFAULT_MARGIN = 0.25          # in within-cohort SD units
N_BOOT = 2000
RNG = np.random.default_rng(20260908)


def load_mutations(study_dir, gene="PTEN", truncating_only=False):
    """Sample barcodes carrying an inactivating mutation in `gene`."""
    path = os.path.join(study_dir, "data_mutations.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    gcol = next((c for c in df.columns if c.lower() == "hugo_symbol"), None)
    scol = next((c for c in df.columns if c.lower() == "tumor_sample_barcode"), None)
    vcol = next((c for c in df.columns if c.lower() == "variant_classification"), None)
    if not all([gcol, scol, vcol]):
        return None
    sub = df[df[gcol].astype(str).str.upper() == gene.upper()]
    keep = TRUNCATING if truncating_only else INACTIVATING
    sub = sub[sub[vcol].astype(str).isin(keep)]
    return set(sub[scol].astype(str))


def _cliffs_delta(a, b):
    """Cliff's delta for a versus b. Positive means a tends to exceed b."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Rank-based computation, O(n log n) rather than the naive pairwise form.
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    return float(2 * u1 / (n1 * n2) - 1)


def _boot_ci(fn, a, b, n_boot=N_BOOT, alpha=0.05):
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = fn(RNG.choice(a, len(a), replace=True),
                     RNG.choice(b, len(b), replace=True))
    vals = vals[np.isfinite(vals)]
    if vals.size < n_boot // 4:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def _tost(a, b, margin):
    """Two one-sided tests for equivalence of means within +/- margin.

    Returns (p_tost, diff, ci90_lo, ci90_hi). Welch standard error is used, so
    unequal group sizes and variances are handled. p_tost is the larger of the
    two one-sided p values; below alpha it supports equivalence.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 5 or n2 < 5:
        return np.nan, np.nan, np.nan, np.nan
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    if se == 0:
        return np.nan, float(diff), np.nan, np.nan
    df = se ** 4 / ((a.var(ddof=1) / n1) ** 2 / (n1 - 1)
                    + (b.var(ddof=1) / n2) ** 2 / (n2 - 1))
    p_lower = stats.t.sf((diff + margin) / se, df)      # H0: diff <= -margin
    p_upper = stats.t.cdf((diff - margin) / se, df)     # H0: diff >= +margin
    p_tost = float(max(p_lower, p_upper))
    crit = stats.t.ppf(0.95, df)
    return p_tost, float(diff), float(diff - crit * se), float(diff + crit * se)


def _contrast(name, a, b, margin):
    """One group comparison, returned as a dict of results."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {f"n_{name}_test": int(a.size), f"n_{name}_ref": int(b.size)}
    if a.size < 5 or b.size < 5:
        out.update({f"p_{name}": None, f"diff_{name}": None,
                    f"delta_{name}": None, f"p_tost_{name}": None})
        return out
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = _cliffs_delta(a, b)
    d_lo, d_hi = _boot_ci(_cliffs_delta, a, b)
    p_tost, diff, ci_lo, ci_hi = _tost(a, b, margin)
    out.update({
        f"mean_{name}_test": round(float(a.mean()), 4),
        f"mean_{name}_ref": round(float(b.mean()), 4),
        f"diff_{name}": round(diff, 4),
        f"ci90_lo_{name}": round(ci_lo, 4),
        f"ci90_hi_{name}": round(ci_hi, 4),
        f"p_{name}": float(p),
        f"delta_{name}": round(delta, 4),
        f"delta_ci_lo_{name}": None if np.isnan(d_lo) else round(d_lo, 4),
        f"delta_ci_hi_{name}": None if np.isnan(d_hi) else round(d_hi, 4),
        f"p_tost_{name}": None if np.isnan(p_tost) else float(p_tost),
    })
    return out


def run(datadir, outdir, gene=GENE_OF_INTEREST, truncating_only=False,
        margin=DEFAULT_MARGIN, exclude_from_pool=("UCEC",)):
    os.makedirs(outdir, exist_ok=True)
    study_dirs = [os.path.join(datadir, d) for d in sorted(os.listdir(datadir))
                  if os.path.isdir(os.path.join(datadir, d))]

    rows = []
    pooled = {"mut_neutral": [], "wt_neutral": [], "hem": [], "hom": [], "study": []}

    for sdir in study_dirs:
        code = os.path.basename(sdir)
        label = A.STUDY_LABELS.get(code, code)
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
        if gene not in expr.index:
            print(f"  [skip] {label}: {gene} absent from expression matrix")
            continue

        mutants = load_mutations(sdir, "PTEN", truncating_only)
        if mutants is None:
            print(f"  [skip] {label}: no mutation file")
            continue

        # Drop samples with no PTEN copy-number call; they are not present
        # in the CNA matrix and cannot be tiered.
        tiers = tiers.dropna()
        cols = list(tiers.index)
        # Within-cohort z-score, matching the Methods.
        vals = np.log2(expr.loc[gene, cols].astype(float).clip(lower=0) + 1)
        z = ((vals - vals.mean()) / vals.std(ddof=0)).values

        is_mut = np.array([c in mutants for c in cols])
        tier_arr = tiers.values

        mut_neutral = z[(tier_arr == "Intact") & is_mut]
        wt_neutral = z[(tier_arr == "Intact") & ~is_mut]
        hem = z[tier_arr == "HemDel"]
        hom = z[tier_arr == "HomDel"]

        row = {"study": label, "gene": gene,
               "truncating_only": truncating_only, "margin_sd": margin}
        row.update(_contrast("mut_vs_wt", mut_neutral, wt_neutral, margin))
        row.update(_contrast("hem_vs_wt", hem, wt_neutral, margin))
        row.update(_contrast("hom_vs_wt", hom, wt_neutral, margin))
        rows.append(row)

        pooled["mut_neutral"].append(mut_neutral)
        pooled["wt_neutral"].append(wt_neutral)
        pooled["hem"].append(hem)
        pooled["hom"].append(hom)
        pooled["study"].append(label)

    res = pd.DataFrame(rows)
    if res.empty:
        print("[mutation_experiment] no cohort produced a usable contrast")
        return res, pd.DataFrame()

    suffix = "_truncating" if truncating_only else ""
    out = os.path.join(outdir, f"S4_mutation_natural_experiment{suffix}.csv")
    res.to_csv(out, index=False)
    print(f"[mutation_experiment] {len(res)} cohorts -> {out}")

    # ---- pooled analyses, with and without each excluded cohort ----------
    pool_rows = []
    labels = pooled["study"]
    for excl in [()] + [(e,) for e in exclude_from_pool]:
        keep = [i for i, l in enumerate(labels) if l not in excl]
        mn = np.concatenate([pooled["mut_neutral"][i] for i in keep]) if keep else np.array([])
        wt = np.concatenate([pooled["wt_neutral"][i] for i in keep]) if keep else np.array([])
        hm = np.concatenate([pooled["hem"][i] for i in keep]) if keep else np.array([])
        ho = np.concatenate([pooled["hom"][i] for i in keep]) if keep else np.array([])
        r = {"excluded": ", ".join(excl) if excl else "none",
             "cohorts": len(keep), "gene": gene,
             "truncating_only": truncating_only, "margin_sd": margin}
        r.update(_contrast("mut_vs_wt", mn, wt, margin))
        r.update(_contrast("hem_vs_wt", hm, wt, margin))
        r.update(_contrast("hom_vs_wt", ho, wt, margin))
        pool_rows.append(r)

    pool = pd.DataFrame(pool_rows)
    pout = os.path.join(outdir, f"S4_mutation_pooled{suffix}.csv")
    pool.to_csv(pout, index=False)
    print(f"[mutation_experiment] pooled results -> {pout}")

    for _, r in pool.iterrows():
        tag = "all cohorts" if r["excluded"] == "none" else f"excluding {r['excluded']}"
        print(f"  {tag}: mutant copy-neutral n={r['n_mut_vs_wt_test']} "
              f"vs wild-type n={r['n_mut_vs_wt_ref']}; "
              f"diff {r['diff_mut_vs_wt']:+.3f} "
              f"(90% CI {r['ci90_lo_mut_vs_wt']:+.3f} to {r['ci90_hi_mut_vs_wt']:+.3f}); "
              f"Mann-Whitney P={r['p_mut_vs_wt']:.3g}; "
              f"TOST P={r['p_tost_mut_vs_wt']:.3g}")

    return res, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--gene", default=GENE_OF_INTEREST)
    ap.add_argument("--truncating-only", action="store_true",
                    help="restrict inactivating calls to truncating variants")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="equivalence margin in within-cohort SD units")
    args = ap.parse_args()
    run(args.datadir, args.outdir, gene=args.gene,
        truncating_only=args.truncating_only, margin=args.margin)


if __name__ == "__main__":
    main()
