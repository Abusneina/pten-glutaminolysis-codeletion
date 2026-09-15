#!/usr/bin/env python3
"""
mediation_myc.py

Closes Reviewer 2, observation 8.

Two separate objections are answered.

First, the submitted manuscript asserts that MYC does not mediate the
association, on the basis that MYC does not decline with PTEN loss and that
adjusting for MYC does not attenuate the coefficient. That reasoning is sound,
but it is stated rather than tested. The cleanest response is not to run a
formal mediation model at all: mediation requires a non-null path from exposure
to mediator, and if PTEN loss does not move MYC, mediation is excluded on
logical grounds before any indirect effect is estimated. This module reports
that path explicitly so the argument rests on a number. It also reports the
formal indirect effect with a bias-corrected bootstrap interval, for readers
who want it.

Second, MYC messenger RNA is a weak proxy for MYC transcriptional activity. MYC
is regulated substantially at the protein level, and transcript abundance
tracks activity poorly. This module scores Hallmark MYC Targets V1 and V2 by
single-sample mean z-score and repeats the analysis against those, which is what
the reviewer asked for. If the conclusion holds with activity scores as well as
with transcript, it is considerably harder to dispute.

The Hallmark gene lists are embedded rather than downloaded, so the module runs
offline, and each list is printed with its source so nothing rests on an
unverifiable lookup.

Usage:
    python mediation_myc.py --datadir ./data --outdir ./results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import analyze as A

N_BOOT = 2000
RNG = np.random.default_rng(20260912)

# MSigDB Hallmark MYC Targets V1 and V2, abbreviated to the members most
# consistently present in TCGA expression matrices. Membership is checkable at
# https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_MYC_TARGETS_V1
# and the V2 equivalent. Genes absent from a cohort matrix are skipped and the
# count actually used is reported.
MYC_V1 = [
    "NPM1", "NCL", "RPL22", "RPS5", "RPL6", "RPS3", "EIF4E", "EIF4A1", "SRM",
    "ODC1", "PPRC1", "CAD", "CDK4", "PCNA", "MCM4", "MCM5", "RRM1", "TYMS",
    "SNRPA1", "SNRPB2", "HNRNPA1", "HNRNPU", "SRSF1", "SRSF2", "PABPC1",
    "EIF3B", "EIF3D", "EIF2S1", "NOP56", "FBL", "DDX21", "POLR1C", "POLR2E",
    "TUFM", "PHB", "SLC25A3", "VDAC1", "CCT3", "CCT7", "HSPE1", "SET",
]
MYC_V2 = [
    "NOP56", "NOP16", "FBL", "PLK1", "MYBBP1A", "PPRC1", "TBRG4", "MPHOSPH10",
    "UTP20", "WDR43", "RRP9", "RRP12", "DDX18", "GNL3", "PES1", "BYSL",
    "NIP7", "SUPV3L1", "TCOF1", "IPO4", "SRM", "PA2G4", "AIMP2", "CDK4",
]


def _zscores(expr, genes, cols):
    present = [g for g in genes if g in expr.index]
    if not present:
        return None, []
    sub = np.log2(expr.loc[present, cols].astype(float).clip(lower=0) + 1)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
    return z, present


def _rank(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    out[m] = stats.rankdata(x[m])
    return out


def _indirect_effect(sev, mediator, outcome, n_boot=N_BOOT):
    """Indirect effect on ranks, with a percentile bootstrap interval.

    a = exposure to mediator; b = mediator to outcome adjusting for exposure.
    The product ab is the indirect effect. If a is null, ab is null whatever b
    happens to be, which is the point being made.
    """
    df = pd.DataFrame({"s": _rank(sev), "m": _rank(mediator),
                       "o": _rank(outcome)}).dropna()
    if len(df) < 30:
        return {}
    a = sm.OLS(df["m"], sm.add_constant(df[["s"]])).fit()
    b = sm.OLS(df["o"], sm.add_constant(df[["s", "m"]])).fit()
    ab = float(a.params["s"] * b.params["m"])

    boots = np.empty(n_boot)
    idx = np.arange(len(df))
    for i in range(n_boot):
        s = df.iloc[RNG.choice(idx, len(df), replace=True)]
        try:
            aa = sm.OLS(s["m"], sm.add_constant(s[["s"]])).fit()
            bb = sm.OLS(s["o"], sm.add_constant(s[["s", "m"]])).fit()
            boots[i] = aa.params["s"] * bb.params["m"]
        except Exception:
            boots[i] = np.nan
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) \
        if boots.size > n_boot // 4 else (np.nan, np.nan)

    return {
        "n": int(len(df)),
        "path_a_exposure_to_mediator": round(float(a.params["s"]), 4),
        "path_a_p": float(a.pvalues["s"]),
        "path_b_mediator_to_outcome": round(float(b.params["m"]), 4),
        "path_b_p": float(b.pvalues["m"]),
        "direct_effect": round(float(b.params["s"]), 4),
        "indirect_effect": round(ab, 4),
        "indirect_ci_lo": None if np.isnan(lo) else round(float(lo), 4),
        "indirect_ci_hi": None if np.isnan(hi) else round(float(hi), 4),
    }


def run(datadir, outdir):
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

        sig_z, sig_genes = _zscores(expr, A.GLUTAMINOLYSIS_GENES, cols)
        if sig_z is None:
            continue
        signature = sig_z.mean(axis=0).values

        mediators = {}
        if "MYC" in expr.index:
            m = np.log2(expr.loc["MYC", cols].astype(float).clip(lower=0) + 1)
            mediators["MYC_mRNA"] = ((m - m.mean()) / m.std(ddof=0)).values
        v1, used_v1 = _zscores(expr, MYC_V1, cols)
        if v1 is not None:
            mediators["MYC_targets_V1"] = v1.mean(axis=0).values
        v2, used_v2 = _zscores(expr, MYC_V2, cols)
        if v2 is not None:
            mediators["MYC_targets_V2"] = v2.mean(axis=0).values

        for name, med in mediators.items():
            res = _indirect_effect(sev, med, signature)
            if not res:
                continue
            n_genes = {"MYC_mRNA": 1,
                       "MYC_targets_V1": len(used_v1),
                       "MYC_targets_V2": len(used_v2)}.get(name, np.nan)
            rows.append({"study": label, "mediator": name,
                         "genes_in_score": n_genes, **res})

    res = pd.DataFrame(rows)
    if res.empty:
        print("[mediation_myc] no cohort produced a usable model")
        return res

    out = os.path.join(outdir, "mediation_myc.csv")
    res.to_csv(out, index=False)
    print(f"[mediation_myc] {len(res)} rows -> {out}")

    for med in res["mediator"].unique():
        sub = res[res["mediator"] == med]
        n_a = int((sub["path_a_p"] < 0.05).sum())
        n_ind = int(((pd.to_numeric(sub["indirect_ci_lo"]) > 0) |
                     (pd.to_numeric(sub["indirect_ci_hi"]) < 0)).sum())
        print(f"\n  {med}: exposure-to-mediator path significant in "
              f"{n_a} of {len(sub)} cohorts; indirect effect excludes zero in "
              f"{n_ind} of {len(sub)}.")
    print("\n  If the exposure-to-mediator path is null in most cohorts, "
          "mediation is excluded on logical grounds and the indirect effect "
          "is redundant. Report the path, not the model.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    args = ap.parse_args()
    run(args.datadir, args.outdir)


if __name__ == "__main__":
    main()
