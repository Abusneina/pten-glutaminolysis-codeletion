#!/usr/bin/env python3
"""
myc_analysis.py
Tests the MYC-axis mechanism for the inverse PTEN-glutaminolysis relationship.

For each included tumor type it asks:
  (1) Does MYC expression correlate with the glutaminolysis signature?
      (Prediction from MYC-driven glutaminolysis: positive.)
  (2) Does MYC expression decline with PTEN loss severity?
      (Prediction from Liu/Muschen B-ALL: negative in cohorts where PTEN
       loss destabilizes MYC.)
  (3) Does MYC statistically account for the PTEN->signature association?
      Compared via the change in the PTEN-severity coefficient on the
      signature before and after adjusting for MYC (a simple mediation check
      using linear models on ranks).

Uses the same loaders, dosage caller, signature, and inclusion rule as
analyze.py. Outputs one CSV and one figure.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import analyze as A  # reuse loaders, dosage, signature, constants

MYC = "MYC"


def zrank(x):
    r = stats.rankdata(x)
    return (r - r.mean()) / r.std(ddof=0)


def run(datadir, outdir, genes=A.GLUTAMINOLYSIS_GENES):
    os.makedirs(outdir, exist_ok=True)
    rows, scatter = [], []
    study_dirs = [os.path.join(datadir, d) for d in sorted(os.listdir(datadir))
                  if os.path.isdir(os.path.join(datadir, d))]
    for sd in study_dirs:
        study = os.path.basename(sd)
        label = A.STUDY_LABELS.get(study, study)
        expr = A.load_expression(sd)
        cna = A.load_cna(sd)
        if MYC not in expr.index:
            continue
        dosage = A.call_pten_dosage(expr, cna)
        score, present = A.signature_score(expr, genes)
        if score is None:
            continue
        common = dosage.dropna().index.intersection(score.index)
        d = dosage[common]
        n = {t: int((d == t).sum()) for t in A.TIERS}
        included = (n["HemDel"] >= A.MIN_TIER and n["HomDel"] >= A.MIN_TIER
                    and n["Intact"] >= A.MIN_INTACT)
        if not included:
            continue
        sev = d.map(A.SEVERITY).astype(float)
        sig = score[common]
        myc = np.log2(expr.loc[MYC, common].astype(float).clip(lower=0) + 1)

        # (1) MYC vs signature
        r_ms, p_ms = stats.spearmanr(myc, sig)
        # (2) MYC vs PTEN severity
        r_mp, p_mp = stats.spearmanr(sev, myc)
        # PTEN severity vs signature (for reference / mediation)
        r_ps, p_ps = stats.spearmanr(sev, sig)

        # (3) crude mediation on ranks: regress sig ~ sev, then sig ~ sev + myc
        Z = pd.DataFrame({"sig": zrank(sig), "sev": zrank(sev), "myc": zrank(myc)})
        m1 = sm.OLS(Z["sig"], sm.add_constant(Z[["sev"]])).fit()
        b_sev_alone = m1.params["sev"]
        m2 = sm.OLS(Z["sig"], sm.add_constant(Z[["sev", "myc"]])).fit()
        b_sev_adj = m2.params["sev"]
        b_myc_adj = m2.params["myc"]
        # proportion of the sev->sig association explained after adding MYC
        prop_attenuation = np.nan
        if b_sev_alone != 0:
            prop_attenuation = (b_sev_alone - b_sev_adj) / b_sev_alone

        cohort = "primary" if label in A.PRIMARY else "extension"
        rows.append({
            "study": label, "cohort": cohort, "n": len(common),
            "rho_MYC_signature": r_ms, "p_MYC_signature": p_ms,
            "rho_severity_MYC": r_mp, "p_severity_MYC": p_mp,
            "rho_severity_signature": r_ps,
            "beta_sev_alone": b_sev_alone, "beta_sev_adj_for_MYC": b_sev_adj,
            "beta_MYC_adj": b_myc_adj,
            "prop_sev_effect_attenuated_by_MYC": prop_attenuation,
        })
        for a_, b_ in zip(myc, sig):
            scatter.append({"study": label, "myc": a_, "sig": b_})

    res = pd.DataFrame(rows).sort_values("rho_MYC_signature", ascending=False)
    res.to_csv(os.path.join(outdir, "myc_mechanism_by_study.csv"), index=False)

    # Figure: two-panel summary
    if len(res):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300)
        order = res.sort_values("rho_MYC_signature")
        y = np.arange(len(order))
        cols = ["#1F3864" if c == "primary" else "#7f8c8d" for c in order["cohort"]]
        axes[0].scatter(order["rho_MYC_signature"], y, c=cols, s=44,
                        edgecolor="black", linewidth=0.5, zorder=3)
        axes[0].axvline(0, color="grey", lw=0.8, ls="--")
        axes[0].set_yticks(y); axes[0].set_yticklabels(order["study"], fontsize=8)
        axes[0].set_xlabel("Spearman rho (MYC vs glutaminolysis signature)")
        axes[0].set_title("MYC tracks the glutaminolysis signature")
        order2 = res.sort_values("rho_severity_MYC")
        y2 = np.arange(len(order2))
        cols2 = ["#1F3864" if c == "primary" else "#7f8c8d" for c in order2["cohort"]]
        axes[1].scatter(order2["rho_severity_MYC"], y2, c=cols2, s=44,
                        edgecolor="black", linewidth=0.5, zorder=3)
        axes[1].axvline(0, color="grey", lw=0.8, ls="--")
        axes[1].set_yticks(y2); axes[1].set_yticklabels(order2["study"], fontsize=8)
        axes[1].set_xlabel("Spearman rho (MYC vs PTEN loss severity)")
        axes[1].set_title("MYC vs PTEN dosage")
        for ax in axes:
            ax.legend(handles=[Patch(facecolor="#1F3864", label="Primary"),
                               Patch(facecolor="#7f8c8d", label="Extension")],
                      loc="lower right", frameon=False, fontsize=8)
        plt.tight_layout(w_pad=3.0); fig.subplots_adjust(left=0.12)
        fig.savefig(os.path.join(outdir, "fig4_myc_mechanism.png"))
        plt.close(fig)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    a = ap.parse_args()
    r = run(a.datadir, a.outdir)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(r[["study","cohort","n","rho_MYC_signature","p_MYC_signature",
             "rho_severity_MYC","p_severity_MYC","rho_severity_signature",
             "prop_sev_effect_attenuated_by_MYC"]].to_string(index=False))
