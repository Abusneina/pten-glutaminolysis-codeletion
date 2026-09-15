#!/usr/bin/env python3
"""
analyze.py
Analysis pipeline for:
"PTEN Copy-Number Dosage and Glutaminolytic Dependency in Human Cancers"

Design follows the dosage logic of Vidotto et al. (Sci Rep 2023;13:5049):
PTEN acts by haploinsufficiency, so one-copy (hemizygous) loss is analyzed
separately from two-copy (homozygous) loss rather than collapsed into a
binary altered/intact call.

Aim 1  Dose-response between PTEN copy-number status (Intact / HemDel /
       HomDel) and a glutaminolysis gene signature, per study.
Aim 3  Prognostic value of PTEN dosage and the glutaminolysis signature
       for overall survival.

Aim 2 (DepMap CRISPR dependency on GLS by PTEN dosage) is in
depmap_module.py.

Input: per-study directories from fetch_data.py (cBioPortal datahub format).
Output: results tables (CSV) and 300-dpi figures under --outdir.
Every statistic is computed from the input data; nothing is hard-coded.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

GLUTAMINOLYSIS_GENES = ["GLS", "SLC1A5", "GOT1", "GLUD1", "GPT2"]
EXTENDED_GENES = ["GLS2", "SLC38A1", "ASNS"]
PTEN = "PTEN"

TIERS = ["Intact", "HemDel", "HomDel"]
SEVERITY = {"Intact": 0, "HemDel": 1, "HomDel": 2}
TIER_COLORS = {"Intact": "#2E9599", "HemDel": "#E0A458", "HomDel": "#B03A2E"}

# Short labels for every TCGA PanCancer Atlas study (code -> abbreviation).
STUDY_LABELS = {
    "acc_tcga_pan_can_atlas_2018": "ACC", "blca_tcga_pan_can_atlas_2018": "BLCA",
    "brca_tcga_pan_can_atlas_2018": "BRCA", "cesc_tcga_pan_can_atlas_2018": "CESC",
    "chol_tcga_pan_can_atlas_2018": "CHOL", "coadread_tcga_pan_can_atlas_2018": "COADREAD",
    "dlbc_tcga_pan_can_atlas_2018": "DLBC", "esca_tcga_pan_can_atlas_2018": "ESCA",
    "gbm_tcga_pan_can_atlas_2018": "GBM", "hnsc_tcga_pan_can_atlas_2018": "HNSC",
    "kich_tcga_pan_can_atlas_2018": "KICH", "kirc_tcga_pan_can_atlas_2018": "KIRC",
    "kirp_tcga_pan_can_atlas_2018": "KIRP", "lgg_tcga_pan_can_atlas_2018": "LGG",
    "lihc_tcga_pan_can_atlas_2018": "LIHC", "luad_tcga_pan_can_atlas_2018": "LUAD",
    "lusc_tcga_pan_can_atlas_2018": "LUSC", "meso_tcga_pan_can_atlas_2018": "MESO",
    "ov_tcga_pan_can_atlas_2018": "OV", "paad_tcga_pan_can_atlas_2018": "PAAD",
    "pcpg_tcga_pan_can_atlas_2018": "PCPG", "prad_tcga_pan_can_atlas_2018": "PRAD",
    "sarc_tcga_pan_can_atlas_2018": "SARC", "skcm_tcga_pan_can_atlas_2018": "SKCM",
    "stad_tcga_pan_can_atlas_2018": "STAD", "tgct_tcga_pan_can_atlas_2018": "TGCT",
    "thca_tcga_pan_can_atlas_2018": "THCA", "thym_tcga_pan_can_atlas_2018": "THYM",
    "ucec_tcga_pan_can_atlas_2018": "UCEC", "ucs_tcga_pan_can_atlas_2018": "UCS",
    "uvm_tcga_pan_can_atlas_2018": "UVM",
}

# Cohorts shown in the survival grid below. A display choice only: every study
# meeting the inclusion rule is analyzed identically and appears in the CSVs.
# These four are the ones the manuscript's survival paragraph discusses, and
# they match the panels of S1 Fig.
KM_DISPLAY = ["UCEC", "GBM", "PRAD", "BRCA"]

# Pre-specified inclusion rule: a study enters the analysis only if it has at
# least MIN_TIER tumors in each of HemDel and HomDel, and at least MIN_INTACT
# intact tumors. This makes "why these tumor types" answerable as "all that
# meet the rule" rather than an arbitrary hand-picked set.
MIN_TIER = 10
MIN_INTACT = 10


def load_expression(study_dir):
    path = os.path.join(study_dir, "data_mrna_seq_v2_rsem.txt")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    df = df[~df[gene_col].isna()].drop_duplicates(subset=gene_col).set_index(gene_col)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id", "Cytoband") if c in df.columns])
    df.columns = [c[:15] for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def load_cna(study_dir):
    path = os.path.join(study_dir, "data_cna.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    df = df.drop_duplicates(subset=gene_col).set_index(gene_col)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id", "Cytoband") if c in df.columns])
    df.columns = [c[:15] for c in df.columns]
    return df


def load_clinical(study_dir):
    p = os.path.join(study_dir, "data_clinical_patient.txt")
    return pd.read_csv(p, sep="\t", comment="#", low_memory=False)


def call_pten_dosage(expr, cna):
    """Return Intact / HemDel / HomDel per sample from GISTIC PTEN calls.

    -2 -> HomDel (two-copy loss); -1 -> HemDel (one-copy loss);
    >=0 -> Intact. Samples without a PTEN CNA value are unassigned (NaN).
    """
    if cna is None or PTEN not in cna.index:
        return pd.Series(index=expr.columns, dtype="object", name="PTEN_dosage")
    row = cna.loc[PTEN]
    out = {}
    for s in expr.columns:
        v = row.get(s, np.nan)
        if pd.isna(v):
            continue
        if v <= -2:
            out[s] = "HomDel"
        elif v == -1:
            out[s] = "HemDel"
        elif v >= 0:
            out[s] = "Intact"
    return pd.Series(out, name="PTEN_dosage").reindex(expr.columns)


def signature_score(expr, genes):
    present = [g for g in genes if g in expr.index]
    if not present:
        return None, present
    sub = np.log2(expr.loc[present].astype(float).clip(lower=0) + 1)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1, ddof=0), axis=0)
    return z.mean(axis=0), present


def aim1(study_dirs, outdir, genes=GLUTAMINOLYSIS_GENES):
    rows, pooled, pair_rows = [], [], []
    for sd in study_dirs:
        study = os.path.basename(sd.rstrip("/"))
        label = STUDY_LABELS.get(study, study)
        expr = load_expression(sd)
        cna = load_cna(sd)
        dosage = call_pten_dosage(expr, cna)
        score, present = signature_score(expr, genes)
        if score is None:
            continue
        common = dosage.dropna().index.intersection(score.index)
        d = dosage[common]
        s = score[common]
        groups = {tier: s[d == tier] for tier in TIERS}
        n = {tier: len(groups[tier]) for tier in TIERS}

        # Pre-specified inclusion rule.
        included = (n["HemDel"] >= MIN_TIER and n["HomDel"] >= MIN_TIER
                    and n["Intact"] >= MIN_INTACT)

        present_tiers = [t for t in TIERS if n[t] >= 3]
        if len(present_tiers) >= 2:
            kw_stat, kw_p = stats.kruskal(*[groups[t] for t in present_tiers])
        else:
            kw_stat, kw_p = np.nan, np.nan
        sev = d.map(SEVERITY).astype(float)
        rho, rho_p = stats.spearmanr(sev.loc[common], s.loc[common])

        rows.append({
            "study": label, "included": included,
            "n_intact": n["Intact"], "n_hemdel": n["HemDel"], "n_homdel": n["HomDel"],
            "median_intact": groups["Intact"].median() if n["Intact"] else np.nan,
            "median_hemdel": groups["HemDel"].median() if n["HemDel"] else np.nan,
            "median_homdel": groups["HomDel"].median() if n["HomDel"] else np.nan,
            "kruskal_H": kw_stat, "kruskal_p": kw_p,
            "spearman_rho_severity": rho, "spearman_p": rho_p,
        })
        if included:
            for tier in TIERS:
                for val in groups[tier]:
                    pooled.append({"study": label, "tier": tier, "score": val})
            for tier in ("HemDel", "HomDel"):
                u, pu = stats.mannwhitneyu(groups[tier], groups["Intact"], alternative="two-sided")
                pair_rows.append({"study": label,
                                  "comparison": f"{tier} vs Intact",
                                  "median_diff": groups[tier].median() - groups["Intact"].median(),
                                  "p_value": pu})

    res = pd.DataFrame(rows)
    # Multiple-testing correction is applied across INCLUDED studies only.
    if len(res):
        inc = res["included"].fillna(False)
        res["kruskal_p_adj_BH"] = np.nan
        res["spearman_p_adj_BH"] = np.nan
        if inc.any():
            res.loc[inc, "kruskal_p_adj_BH"] = multipletests(
                res.loc[inc, "kruskal_p"].fillna(1.0), method="fdr_bh")[1]
            res.loc[inc, "spearman_p_adj_BH"] = multipletests(
                res.loc[inc, "spearman_p"].fillna(1.0), method="fdr_bh")[1]
    pair = pd.DataFrame(pair_rows)
    if len(pair):
        pair["p_adj_BH"] = multipletests(pair["p_value"], method="fdr_bh")[1]
    res.to_csv(os.path.join(outdir, "aim1_dose_response_by_study.csv"), index=False)
    pair.to_csv(os.path.join(outdir, "aim1_pairwise_vs_intact.csv"), index=False)

    _fig1_signature_boxplot(pd.DataFrame(pooled), outdir)
    _fig1b_trend_forest(res, outdir)
    return res, pair


def _fig1_signature_boxplot(pooled, outdir):
    """Boxplot of signature by dosage across every included study."""
    if len(pooled) == 0:
        return
    studies = sorted(set(pooled["study"]))
    fig, ax = plt.subplots(figsize=(1.25 * len(studies) + 2, 4.4), dpi=300)
    ticks, labels = [], []
    for i, st in enumerate(studies):
        for j, tier in enumerate(TIERS):
            vals = pooled[(pooled.study == st) & (pooled.tier == tier)]["score"].dropna()
            if len(vals) == 0:
                continue
            bp = ax.boxplot(vals, positions=[i * 4 + j], widths=0.8,
                            patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor=TIER_COLORS[tier], alpha=0.8, edgecolor="black")
            for med in bp["medians"]:
                med.set(color="black")
        ticks.append(i * 4 + 1)
        labels.append(st)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Glutaminolysis signature score (mean gene z-score)")
    ax.set_title("Glutaminolysis signature across PTEN dosage, included studies", pad=26)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.legend(handles=[Patch(facecolor=TIER_COLORS[t], label=t) for t in TIERS],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_signature_by_pten_dosage.png"))
    plt.close(fig)


def _fig1b_trend_forest(res, outdir):
    """Pan-cancer forest of the Spearman dose-response across included studies."""
    if len(res) == 0:
        return
    d = res[res["included"] == True].copy()
    if len(d) == 0:
        return
    d = d.sort_values("spearman_rho_severity", ascending=True)
    y = np.arange(len(d))
    sig = d["spearman_p_adj_BH"] < 0.05
    colors = ["#7B2D26" if s else "#6E6E6E" for s in sig]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(d) + 1.8), dpi=300)
    ax.scatter(d["spearman_rho_severity"], y, c=colors,
               s=[46 if s else 26 for s in sig],
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}{' *' if s else ''}" for lab, s in zip(d['study'], sig)], fontsize=8)
    ax.set_xlabel("Spearman rho (glutaminolysis signature vs PTEN loss severity)")
    ax.set_title("Pan-cancer dose-response across included tumor types", pad=14)
    ax.legend(handles=[Patch(facecolor="#7B2D26", label="Significant"),
                       Patch(facecolor="#6E6E6E", label="Not significant")],
              loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    ax.margins(y=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1b_pancancer_trend_forest.png"))
    plt.close(fig)


def aim3(study_dirs, outdir, genes=GLUTAMINOLYSIS_GENES):
    surv_rows, km_records = [], []
    for sd in study_dirs:
        study = os.path.basename(sd.rstrip("/"))
        label = STUDY_LABELS.get(study, study)
        expr = load_expression(sd)
        cna = load_cna(sd)
        clin = load_clinical(sd)
        dosage = call_pten_dosage(expr, cna)
        score, _ = signature_score(expr, genes)
        if score is None:
            continue
        os_time = next((c for c in clin.columns if c.upper() == "OS_MONTHS"), None)
        os_stat = next((c for c in clin.columns if c.upper() == "OS_STATUS"), None)
        pid = next((c for c in clin.columns if c.upper().replace("#", "") == "PATIENT_ID"),
                   clin.columns[0])
        if os_time is None or os_stat is None:
            continue
        clin = clin[[pid, os_time, os_stat]].copy()
        clin["pid"] = clin[pid].astype(str).str[:12]
        clin["time"] = pd.to_numeric(clin[os_time], errors="coerce")
        clin["event"] = clin[os_stat].astype(str).str.contains(
            "1|DECEASED", case=False, regex=True).astype(int)

        df = pd.DataFrame({"sample": score.index})
        df["pid"] = df["sample"].str[:12]
        df["sig"] = score.values
        df["tier"] = dosage.reindex(df["sample"]).values
        df = df.dropna(subset=["tier"])
        df = df.merge(clin[["pid", "time", "event"]], on="pid", how="inner").dropna(subset=["time"])
        df = df[df["time"] > 0]
        if df["pid"].duplicated().any():
            df = df.sort_values("sample").drop_duplicates("pid")
        if len(df) < 20:
            continue

        n_int = int((df.tier == "Intact").sum())
        n_hem = int((df.tier == "HemDel").sum())
        n_hom = int((df.tier == "HomDel").sum())
        included = (n_hem >= MIN_TIER and n_hom >= MIN_TIER and n_int >= MIN_INTACT)

        df["severity"] = df["tier"].map(SEVERITY)
        try:
            cph = CoxPHFitter().fit(df[["time", "event", "severity", "sig"]], "time", "event")
            hr_sev = float(np.exp(cph.params_["severity"]))
            p_sev = float(cph.summary.loc["severity", "p"])
            hr_sig = float(np.exp(cph.params_["sig"]))
            p_sig = float(cph.summary.loc["sig", "p"])
        except Exception:
            hr_sev = p_sev = hr_sig = p_sig = np.nan
        try:
            lr = multivariate_logrank_test(df["time"], df["tier"], df["event"])
            p_lr = float(lr.p_value)
        except Exception:
            p_lr = np.nan

        surv_rows.append({
            "study": label, "included": included, "n": len(df),
            "n_intact": n_int, "n_hemdel": n_hem, "n_homdel": n_hom,
            "HR_per_severity_step": hr_sev, "p_severity": p_sev,
            "HR_signature": hr_sig, "p_signature": p_sig,
            "multivariate_logrank_p": p_lr,
        })
        if included:
            km_records.append((label, df))

    res = pd.DataFrame(surv_rows)
    res.to_csv(os.path.join(outdir, "aim3_survival_by_dosage.csv"), index=False)

    # KM grid restricted to KM_DISPLAY to stay legible; every included study is
    # still modeled in the CSV above.
    prim = [(lab, df) for (lab, df) in km_records if lab in KM_DISPLAY]
    prim = sorted(prim, key=lambda x: KM_DISPLAY.index(x[0]))
    if prim:
        nfig = len(prim)
        fig, axes = plt.subplots(1, nfig, figsize=(4.9 * nfig, 4.0), dpi=300, squeeze=False)
        for ax, (label, df) in zip(axes[0], prim):
            kmf = KaplanMeierFitter()
            for tier in TIERS:
                sub = df[df.tier == tier]
                if len(sub) >= 5:
                    kmf.fit(sub["time"], sub["event"], label=f"{tier} (n={len(sub)})")
                    kmf.plot_survival_function(ax=ax, color=TIER_COLORS[tier], ci_show=False)
            ax.set_title(label)
            ax.set_xlabel("Overall survival (months)")
            ax.set_ylabel("Survival probability")
            ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", loc="upper right")
        plt.tight_layout(w_pad=2.0)
        fig.savefig(os.path.join(outdir, "fig3_km_by_pten_dosage.png"))
        plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--extended", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    genes = GLUTAMINOLYSIS_GENES + (EXTENDED_GENES if args.extended else [])
    study_dirs = [os.path.join(args.datadir, d) for d in sorted(os.listdir(args.datadir))
                  if os.path.isdir(os.path.join(args.datadir, d))]
    print(f"[analyze] {len(study_dirs)} study folders found")
    print(f"[analyze] inclusion rule: HemDel>={MIN_TIER}, HomDel>={MIN_TIER}, "
          f"Intact>={MIN_INTACT} tumors")
    r1, pair = aim1(study_dirs, args.outdir, genes)
    if len(r1):
        n_inc = int(r1["included"].sum())
        print(f"\n[analyze] {n_inc}/{len(r1)} studies met the inclusion rule")
        print("=== Aim 1: dose-response (included studies) ===")
        cols = ["study", "n_intact", "n_hemdel", "n_homdel",
                "spearman_rho_severity", "spearman_p_adj_BH", "kruskal_p_adj_BH"]
        print(r1[r1["included"]][cols].to_string(index=False))
        excl = r1[~r1["included"]]["study"].tolist()
        if excl:
            print(f"\n[analyze] excluded (insufficient tier counts): {', '.join(excl)}")
    else:
        print("(no Aim 1 results)")
    r3 = aim3(study_dirs, args.outdir, genes)
    print("\n=== Aim 3: survival by PTEN dosage (included studies) ===")
    if len(r3):
        print(r3[r3["included"]][["study", "n", "HR_per_severity_step",
                                  "p_severity", "multivariate_logrank_p"]].to_string(index=False))
    else:
        print("(no Aim 3 results)")
    print(f"\n[analyze] outputs -> {args.outdir}")


if __name__ == "__main__":
    main()
