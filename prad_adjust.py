#!/usr/bin/env python3
"""
prad_adjust.py

Analysis F. Tests whether the decline of the off-chromosome-10 signature genes
in prostate carcinoma survives adjustment for the clinical and compositional
features that co-vary with PTEN deletion in that tissue.

Why this is needed. The manuscript reads the prostate decline as evidence of a
preserved signaling component. It cannot be, because the signaling model
predicts an increase and the observed effect is a decrease. What the decline
does need is an explanation, and prostate carcinoma is the lineage where PTEN
deletion travels with several other things: higher grade, TMPRSS2-ERG fusion,
greater proliferative activity, and differences in tumour cellularity. Any of
those could move a bulk expression score without a metabolic program.

Two questions are answered:

  F1. Does the off-chromosome-10 decline in prostate persist once those
      covariates are held constant?
  F2. Do the positive off-chromosome-10 coefficients in the other cohorts
      survive correction for multiple testing? If they do not, then no lineage
      shows a credible effect in either direction, which is the cleanest
      statement the data support.

Covariates are discovered rather than assumed. The module reports exactly which
were found in the cohort files and which were not, so nothing silently drops
out of the model. Two are derived from expression when no clinical column
exists: ERG expression as a proxy for TMPRSS2-ERG fusion status, and a
proliferation index from a standard set of cell-cycle genes. Both are stated in
the output so they can be described honestly in the Methods.

Usage:
    python prad_adjust.py --datadir ./data --outdir ./results
    python prad_adjust.py --datadir ./data --outdir ./results --cohort PRAD
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import analyze as A

OFF_CHR10 = ["GLS", "SLC1A5", "GPT2"]
CHR10 = ["GLUD1", "GOT1"]

# Standard proliferation genes. Used only to build an index when no clinical
# proliferation measure exists; every gene is reported in the output.
PROLIFERATION = ["MKI67", "TOP2A", "CCNB1", "BUB1", "PLK1", "AURKA",
                 "CDK1", "RRM2", "TYMS", "PCNA", "CCNA2", "MCM2"]

ERG_PROXY = "ERG"

# Candidate clinical column names, matched case-insensitively after
# normalization. cBioPortal naming is not consistent across studies.
PURITY_COLS = ["TUMOR_PURITY", "PURITY", "ABSOLUTE_PURITY", "ESTIMATE_PURITY"]
GRADE_COLS = ["GLEASON_SCORE", "GLEASON_GRADE_GROUP", "GRADE_GROUP",
              "REVIEWED_GLEASON_SUM", "GLEASON_PATTERN_PRIMARY", "GRADE"]
# Pathological T stage, used when no numeric grade is recorded. In the TCGA
# prostate cohort GRADE and AJCC_PATHOLOGIC_TUMOR_STAGE are entirely empty
# while PATH_T_STAGE is populated for 487 of 494 samples.
STAGE_T_COLS = ["PATH_T_STAGE", "CLIN_T_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE"]


def _stage_to_ordinal(v):
    """Map T-stage strings to an ordinal scale: T2A < T2B < T2C < T3A < T3B < T4."""
    if not isinstance(v, str):
        return np.nan
    s = v.strip().upper().replace("STAGE", "").strip()
    if not s.startswith("T"):
        return np.nan
    digits = "".join(ch for ch in s[1:] if ch.isdigit())
    if not digits:
        return np.nan
    major = int(digits[0])
    letter = next((ch for ch in s[1:] if ch.isalpha()), "")
    minor = {"A": 0, "B": 1, "C": 2}.get(letter, 0)
    return float(major) + minor / 3.0
STAGE_COLS = ["AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "CLIN_T_STAGE"]

N_BOOT = 1000
RNG = np.random.default_rng(20260908)


def _norm(c):
    return str(c).strip().upper().replace(" ", "_").lstrip("#")


def _read_clinical(study_dir, fname):
    path = os.path.join(study_dir, fname)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = [_norm(c) for c in df.columns]
    return df


def _find_col(df, candidates):
    if df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _zscore(v):
    v = pd.to_numeric(v, errors="coerce").astype(float)
    sd = v.std(ddof=0)
    return (v - v.mean()) / sd if sd and np.isfinite(sd) else v * np.nan


def _composite(expr, genes, cols):
    present = [g for g in genes if g in expr.index]
    if not present:
        return None, []
    sub = np.log2(expr.loc[present, cols].astype(float).clip(lower=0) + 1)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
    return z.mean(axis=0), present


def _rank(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    out[m] = stats.rankdata(x[m])
    return out


def build_covariates(study_dir, expr, cols):
    """Assemble the covariate frame, reporting provenance for each column."""
    cov = pd.DataFrame(index=cols)
    provenance = []

    stab = _read_clinical(study_dir, "data_clinical_sample.txt")
    ptab = _read_clinical(study_dir, "data_clinical_patient.txt")

    if not stab.empty and "SAMPLE_ID" in stab.columns:
        stab["SAMPLE_ID"] = stab["SAMPLE_ID"].astype(str).str[:15]
        stab = stab.drop_duplicates("SAMPLE_ID").set_index("SAMPLE_ID")
    if not ptab.empty and "PATIENT_ID" in ptab.columns:
        ptab = ptab.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    patients = pd.Series([c[:12] for c in cols], index=cols)

    # --- tumour purity ---------------------------------------------------
    pcol = _find_col(stab, PURITY_COLS) or _find_col(ptab, PURITY_COLS)
    if pcol and not stab.empty and pcol in stab.columns:
        cov["purity"] = pd.to_numeric(stab[pcol], errors="coerce").reindex(cols).values
        provenance.append(("purity", f"clinical sample column {pcol}"))
    elif pcol and not ptab.empty and pcol in ptab.columns:
        cov["purity"] = pd.to_numeric(ptab[pcol], errors="coerce").reindex(patients).values
        provenance.append(("purity", f"clinical patient column {pcol}"))
    else:
        provenance.append(("purity", "NOT AVAILABLE in cohort clinical files"))

    # --- grade, or pathological T stage when no numeric grade exists -----
    grade_vals, grade_src = None, None
    gcol = _find_col(ptab, GRADE_COLS) or _find_col(stab, GRADE_COLS)
    if gcol and not ptab.empty and gcol in ptab.columns:
        v = pd.to_numeric(ptab[gcol], errors="coerce").reindex(patients)
        if v.notna().sum() >= 30:
            grade_vals, grade_src = v.values, f"clinical patient column {gcol}"
    if grade_vals is None and gcol and not stab.empty and gcol in stab.columns:
        v = pd.to_numeric(stab[gcol], errors="coerce").reindex(cols)
        if v.notna().sum() >= 30:
            grade_vals, grade_src = v.values, f"clinical sample column {gcol}"

    if grade_vals is None:
        # A grade column may exist but be entirely empty, as GRADE is in the
        # TCGA prostate cohort. Fall through to T stage, which is ordinal.
        scol = _find_col(ptab, STAGE_T_COLS) or _find_col(stab, STAGE_T_COLS)
        for tbl, idx, name in ((ptab, patients, "clinical patient"),
                               (stab, cols, "clinical sample")):
            if grade_vals is not None or not scol or tbl.empty or scol not in tbl.columns:
                continue
            v = tbl[scol].reindex(idx).map(_stage_to_ordinal)
            if v.notna().sum() >= 30:
                grade_vals = v.values
                grade_src = (f"{name} column {scol}, mapped to an ordinal "
                             f"T-stage scale; no numeric grade recorded")

    if grade_vals is not None:
        cov["grade"] = grade_vals
        provenance.append(("grade", grade_src))
    else:
        provenance.append(("grade", "NOT AVAILABLE in cohort clinical files"))

    # --- ERG, proxy for TMPRSS2-ERG fusion -------------------------------
    if ERG_PROXY in expr.index:
        erg = np.log2(expr.loc[ERG_PROXY, cols].astype(float).clip(lower=0) + 1)
        cov["erg"] = _zscore(erg).values
        provenance.append(("erg", "ERG expression z-score, proxy for TMPRSS2-ERG "
                                  "fusion status; no fusion call in the cohort files"))
    else:
        provenance.append(("erg", "NOT AVAILABLE: ERG absent from expression matrix"))

    # --- proliferation index ---------------------------------------------
    prolif, used = _composite(expr, PROLIFERATION, cols)
    if prolif is not None:
        cov["proliferation"] = prolif.values
        provenance.append(("proliferation",
                           "mean z-score of " + ", ".join(used)))
    else:
        provenance.append(("proliferation", "NOT AVAILABLE"))

    return cov, provenance


def _fit(y, sev, cov):
    """Rank-based OLS of y on severity, with and without covariates.

    Ranks are used because the underlying tests in the manuscript are Spearman,
    so this keeps the adjusted and unadjusted estimates on the same footing.
    Returns a dict of coefficients, p values and the sample size actually used.
    """
    ry, rsev = _rank(y), _rank(sev)
    base = pd.DataFrame({"y": ry, "sev": rsev})
    for c in cov.columns:
        base[c] = _rank(cov[c].values)

    out = {}
    crude = base[["y", "sev"]].dropna()
    if len(crude) < 30:
        return None
    X = sm.add_constant(crude[["sev"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = sm.OLS(crude["y"], X).fit()
    sd_ratio = crude["sev"].std(ddof=1) / crude["y"].std(ddof=1)
    out["n_crude"] = int(len(crude))
    out["beta_crude_std"] = float(m0.params["sev"] * sd_ratio)
    out["p_crude"] = float(m0.pvalues["sev"])

    usable = [c for c in cov.columns if base[c].notna().sum() >= 30]
    dropped = [c for c in cov.columns if c not in usable]
    full = base[["y", "sev"] + usable].dropna()
    if len(full) < 30 or not usable:
        out["covariates_used"] = ""
        out["n_adjusted"] = 0
        out["beta_adjusted_std"] = np.nan
        out["p_adjusted"] = np.nan
        return out
    X = sm.add_constant(full[["sev"] + usable])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = sm.OLS(full["y"], X).fit()
    sd_ratio = full["sev"].std(ddof=1) / full["y"].std(ddof=1)
    out["covariates_used"] = ", ".join(usable)
    out["n_adjusted"] = int(len(full))
    out["beta_adjusted_std"] = float(m1.params["sev"] * sd_ratio)
    out["p_adjusted"] = float(m1.pvalues["sev"])
    # Attenuation: how much of the crude association the covariates absorb.
    if out["beta_crude_std"] != 0:
        out["attenuation_pct"] = round(
            100 * (1 - out["beta_adjusted_std"] / out["beta_crude_std"]), 1)
    else:
        out["attenuation_pct"] = np.nan
    return out


def run_cohort(study_dir, label, outdir):
    expr = A.load_expression(study_dir)
    cna = A.load_cna(study_dir)
    tiers = A.call_pten_dosage(expr, cna)
    if tiers is None or not len(tiers.dropna()):
        return None
    tiers = tiers.dropna()
    cols = list(tiers.index)
    sev = tiers.map(A.SEVERITY).astype(float).values

    cov, provenance = build_covariates(study_dir, expr, cols)

    print(f"\nCovariates for {label}:")
    for name, source in provenance:
        mark = "  " if "NOT AVAILABLE" not in source else "! "
        print(f"  {mark}{name}: {source}")

    rows = []
    targets = {"off_chr10_composite": OFF_CHR10, "chr10_composite": CHR10}
    for name, genes in targets.items():
        comp, used = _composite(expr, genes, cols)
        if comp is None:
            continue
        res = _fit(comp.values, sev, cov)
        if res:
            rows.append({"cohort": label, "target": name,
                         "genes": ", ".join(used), **res})
    for g in OFF_CHR10 + CHR10:
        if g not in expr.index:
            continue
        v = np.log2(expr.loc[g, cols].astype(float).clip(lower=0) + 1)
        res = _fit(v.values, sev, cov)
        if res:
            rows.append({"cohort": label, "target": g, "genes": g, **res})

    df = pd.DataFrame(rows)
    prov = pd.DataFrame(provenance, columns=["covariate", "source"])
    prov.insert(0, "cohort", label)
    return df, prov


def run_all_cohorts_offchr10(datadir):
    """F2: off-chromosome-10 association per cohort, with BH correction."""
    rows = []
    for d in sorted(os.listdir(datadir)):
        sdir = os.path.join(datadir, d)
        if not os.path.isdir(sdir):
            continue
        label = A.STUDY_LABELS.get(d, d)
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
        sev = tiers.map(A.SEVERITY).astype(float).values
        comp, used = _composite(expr, OFF_CHR10, cols)
        if comp is None:
            continue
        print(f"  [F2] {label}: {len(cols)} tumors", flush=True)
        rho, p = stats.spearmanr(sev, comp.values, nan_policy="omit")
        rows.append({"cohort": label, "n": len(cols),
                     "rho_off_chr10": round(float(rho), 4), "p_raw": float(p)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("rho_off_chr10")
    df["p_bh"] = multipletests(df["p_raw"], method="fdr_bh")[1]
    df["significant_bh"] = df["p_bh"] < 0.05
    df["direction"] = np.where(df["rho_off_chr10"] > 0,
                               "consistent with signaling",
                               "opposite to signaling")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--cohort", default="PRAD",
                    help="cohort label for the covariate-adjusted analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- F1 ---------------------------------------------------------------
    target_dir = None
    for d in sorted(os.listdir(args.datadir)):
        if A.STUDY_LABELS.get(d, d) == args.cohort:
            target_dir = os.path.join(args.datadir, d)
            break
    if target_dir is None:
        print(f"[F1] cohort {args.cohort} not found in {args.datadir}")
    else:
        out = run_cohort(target_dir, args.cohort, args.outdir)
        if out is not None:
            df, prov = out
            df.to_csv(os.path.join(args.outdir,
                      f"F1_{args.cohort}_covariate_adjusted.csv"), index=False)
            prov.to_csv(os.path.join(args.outdir,
                        f"F1_{args.cohort}_covariate_provenance.csv"), index=False)
            print(f"\n[F1] {args.cohort} results -> {args.outdir}")
            show = df[df["target"].isin(["off_chr10_composite", "chr10_composite"])]
            for _, r in show.iterrows():
                print(f"  {r['target']}: crude beta {r['beta_crude_std']:+.3f} "
                      f"(P={r['p_crude']:.3g}, n={r['n_crude']}) -> "
                      f"adjusted {r['beta_adjusted_std']:+.3f} "
                      f"(P={r['p_adjusted']:.3g}, n={r['n_adjusted']}); "
                      f"attenuation {r.get('attenuation_pct', float('nan'))}%")
            print("  Adjusting for covariates that co-vary with PTEN deletion is "
                  "not the same as explaining the effect away; read the "
                  "attenuation alongside the covariate provenance file.")

    # --- F2 ---------------------------------------------------------------
    f2 = run_all_cohorts_offchr10(args.datadir)
    if not f2.empty:
        f2.to_csv(os.path.join(args.outdir, "F2_offchr10_multiple_testing.csv"),
                  index=False)
        print(f"\n[F2] off-chromosome-10 association by cohort -> {args.outdir}")
        print(f2.to_string(index=False))
        n_sig = int(f2["significant_bh"].sum())
        n_pos = int((f2["rho_off_chr10"] > 0).sum())
        print(f"\n  {n_pos} of {len(f2)} cohorts positive; "
              f"{n_sig} significant after Benjamini-Hochberg correction.")
        if n_sig == 0:
            print("  No cohort shows a credible off-chromosome-10 effect in "
                  "either direction. That is the cleanest statement the data "
                  "support and should be what the manuscript says.")


if __name__ == "__main__":
    main()
