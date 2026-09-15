#!/usr/bin/env python3
"""
aneuploidy_partial.py

Reconstructs the aneuploidy-adjusted analysis reported as S3 Table: the
Spearman partial correlation between PTEN loss severity and each signature
gene's expression, controlling for the per-sample aneuploidy score.

This analysis is described in the Methods of PONE-D-26-35226 but was not
present in the v1.0 code deposit. The numbers it produces must be checked
against the published S3 Table before the revision is submitted.

Two things are computed that the original analysis did not have:

  * A gene-level copy-number adjustment. A genome-wide aneuploidy score cannot
    isolate 10q loss, because a tumor with a focal 10q deletion and an
    otherwise quiet genome carries a low score. Conditioning on the gene's own
    copy number is the direct test, and is what Reviewer 2 asked for.
  * Bootstrap confidence intervals on every coefficient, so that trivial
    effects are visible as trivial rather than being read as part of a pattern.

Partial Spearman correlation is computed on ranks by the standard one-covariate
formula, with significance from the t statistic on n - 3 degrees of freedom.
This matches pingouin's partial_corr(method='spearman') to floating-point
tolerance; a check against pingouin is run when that package is installed.

Usage:
    python3 aneuploidy_partial.py --datadir ./data --outdir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A

ANEUPLOIDY_COLS = ("ANEUPLOIDY_SCORE", "ANEUPLOIDY SCORE", "AneuploidyScore")
N_BOOT = 2000
RNG = np.random.default_rng(20260908)


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z.

    Returns (rho, p, n). Ranks are taken first, then the standard
    one-covariate partial correlation formula is applied to the ranks.
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = x.size
    if n < 10:
        return np.nan, np.nan, n
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return np.nan, np.nan, n
    rho = (rxy - rxz * ryz) / denom
    rho = float(np.clip(rho, -1.0, 1.0))
    if abs(rho) >= 1.0 or n <= 3:
        return rho, 0.0, n
    t = rho * np.sqrt((n - 3) / (1 - rho ** 2))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return rho, float(p), n


def _boot_ci(fn, arrays, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI for a statistic computed on paired arrays."""
    n = arrays[0].size
    if n < 10:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals[b] = fn(*[a[idx] for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size < n_boot // 4:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def _load_aneuploidy(study_dir):
    """Per-sample aneuploidy score, or None when the cohort has no column."""
    path = os.path.join(study_dir, "data_clinical_sample.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    sid = next((c for c in df.columns if c.upper().replace("#", "") == "SAMPLE_ID"), None)
    if sid is None:
        return None
    col = next((c for c in df.columns
                if c.upper().replace(" ", "_") in
                [a.upper().replace(" ", "_") for a in ANEUPLOIDY_COLS]), None)
    if col is None:
        return None
    out = pd.to_numeric(df.set_index(sid)[col], errors="coerce").dropna()
    return out if len(out) else None


def _gene_cn(cna, gene, cols):
    if gene not in cna.index:
        return None
    return pd.to_numeric(cna.loc[gene, cols], errors="coerce")


def run(datadir, outdir, genes=None):
    genes = genes or A.GLUTAMINOLYSIS_GENES
    os.makedirs(outdir, exist_ok=True)

    study_dirs = [os.path.join(datadir, d) for d in sorted(os.listdir(datadir))
                  if os.path.isdir(os.path.join(datadir, d))]

    rows = []
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

        # Samples with no PTEN copy-number call carry a NaN tier and are
        # absent from the CNA matrix entirely, so they must go before any
        # gene is looked up by column.
        tiers = tiers.dropna()
        cols = list(tiers.index)
        sev = tiers.map(A.SEVERITY).astype(float).values

        aneu_series = _load_aneuploidy(sdir)
        if aneu_series is not None:
            aneu = pd.Series(aneu_series).reindex(cols).astype(float).values
            has_aneu = np.isfinite(aneu).sum() >= 10
        else:
            aneu, has_aneu = np.full(len(cols), np.nan), False

        logexpr = np.log2(expr[cols].astype(float).clip(lower=0) + 1)

        for g in genes:
            if g not in logexpr.index:
                continue
            y = logexpr.loc[g].values.astype(float)

            raw_rho, raw_p = stats.spearmanr(sev, y, nan_policy="omit")
            lo, hi = _boot_ci(lambda a, b: stats.spearmanr(a, b).statistic, (sev, y))

            if has_aneu:
                a_rho, a_p, a_n = _partial_spearman(sev, y, aneu)
                a_lo, a_hi = _boot_ci(
                    lambda a, b, c: _partial_spearman(a, b, c)[0], (sev, y, aneu))
            else:
                a_rho = a_p = np.nan
                a_n = int(np.isfinite(y).sum())
                a_lo = a_hi = np.nan

            gcn = _gene_cn(cna, g, cols)
            if gcn is not None and gcn.notna().sum() >= 10:
                c_rho, c_p, _ = _partial_spearman(sev, y, gcn.values.astype(float))
                c_lo, c_hi = _boot_ci(
                    lambda a, b, c: _partial_spearman(a, b, c)[0],
                    (sev, y, gcn.values.astype(float)))
            else:
                c_rho = c_p = c_lo = c_hi = np.nan

            rows.append({
                "study": label,
                "gene": g,
                "n": int(np.isfinite(y).sum()),
                "rho_unadjusted": round(float(raw_rho), 4),
                "p_unadjusted": float(raw_p),
                "ci_lo_unadjusted": None if np.isnan(lo) else round(lo, 4),
                "ci_hi_unadjusted": None if np.isnan(hi) else round(hi, 4),
                "aneuploidy_available": has_aneu,
                "rho_partial_aneuploidy": None if np.isnan(a_rho) else round(a_rho, 4),
                "p_partial_aneuploidy": None if np.isnan(a_p) else float(a_p),
                "ci_lo_partial_aneuploidy": None if np.isnan(a_lo) else round(a_lo, 4),
                "ci_hi_partial_aneuploidy": None if np.isnan(a_hi) else round(a_hi, 4),
                "rho_partial_own_cn": None if np.isnan(c_rho) else round(c_rho, 4),
                "p_partial_own_cn": None if np.isnan(c_p) else float(c_p),
                "ci_lo_partial_own_cn": None if np.isnan(c_lo) else round(c_lo, 4),
                "ci_hi_partial_own_cn": None if np.isnan(c_hi) else round(c_hi, 4),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[aneuploidy_partial] no cohort met the inclusion rule")
        return res

    out = os.path.join(outdir, "S3_aneuploidy_partial_correlation.csv")
    res.to_csv(out, index=False)
    print(f"[aneuploidy_partial] {len(res)} gene-cohort rows -> {out}")

    missing = sorted(res.loc[~res["aneuploidy_available"], "study"].unique())
    if missing:
        print(f"  cohorts without an aneuploidy score: {', '.join(missing)}")

    # Headline contrast: does conditioning on the gene's own copy number remove
    # the association for the chromosome-10 genes but leave the others alone?
    chr10 = res[res["gene"].isin(["GLUD1", "GOT1"])]
    other = res[~res["gene"].isin(["GLUD1", "GOT1"])]
    for name, sub in (("chr10 (GLUD1, GOT1)", chr10), ("off-chr10", other)):
        if sub.empty:
            continue
        print(f"  {name}: mean rho unadjusted {sub['rho_unadjusted'].mean():+.3f}; "
              f"aneuploidy-adjusted {pd.to_numeric(sub['rho_partial_aneuploidy']).mean():+.3f}; "
              f"own-CN-adjusted {pd.to_numeric(sub['rho_partial_own_cn']).mean():+.3f}")
    return res


def _check_against_pingouin():
    """Verify the partial correlation implementation if pingouin is present."""
    try:
        import pingouin as pg
    except ImportError:
        print("[check] pingouin not installed; internal implementation used")
        return
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = 0.6 * z + rng.normal(size=300)
    y = -0.4 * z + 0.3 * x + rng.normal(size=300)
    mine = _partial_spearman(x, y, z)[0]
    theirs = pg.partial_corr(
        data=pd.DataFrame({"x": x, "y": y, "z": z}),
        x="x", y="y", covar="z", method="spearman")["r"].iloc[0]
    print(f"[check] partial rho: internal {mine:.6f} vs pingouin {theirs:.6f}")
    assert abs(mine - theirs) < 1e-6, "partial correlation implementation differs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--check", action="store_true",
                    help="verify the partial correlation against pingouin and exit")
    args = ap.parse_args()
    if args.check:
        _check_against_pingouin()
        return
    run(args.datadir, args.outdir)


if __name__ == "__main__":
    main()
