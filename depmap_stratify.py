#!/usr/bin/env python3
"""
depmap_stratify.py

Closes Reviewer 1, point 3, and Reviewer 3's caveat on the DepMap null result.

Two objections, one module.

Reviewer 1 asks whether GLS dependency was examined in cell lines that actually
resemble the tumors analysed, matching both the PTEN expression thresholds and
the 10q copy-number losses seen in bulk TCGA. A null result across all cell
lines is weaker than a null result in the lines that mirror the tumor
population, because most DepMap lines are not PTEN-deleted at all.

Reviewer 3 points out that GLS is a weak dependency in most cell lines
regardless of PTEN status, so a null difference across dosage tiers is close to
uninformative unless the dynamic range is shown. If almost no line depends on
GLS, there is nothing for PTEN status to modify.

This module therefore reports three things the submitted analysis did not:

  1. GLS dependency stratified jointly by PTEN copy number and PTEN expression,
     rather than by copy number alone, so lines matching the bulk tumor profile
     can be examined separately.
  2. The dependency distribution of GLS against reference genes, so the reader
     can see whether GLS is a dependency at all in this panel. A common
     essential gene and a non-expressed control are used as anchors.
  3. An equivalence test on the tier comparison, so "does not differ" becomes a
     claim that can fail rather than an absence of significance.

The bounding number to report is the proportion of lines with a GLS gene effect
below the conventional dependency threshold of -0.5. If that proportion is very
small, say so plainly: the null result then constrains little, and the
transcriptional and copy-number evidence has to carry the argument alone.

DepMap files are not reachable from the analysis allowlist and must be
downloaded from https://depmap.org/portal/download/ (free account):
  CRISPRGeneEffect.csv, OmicsCNGene.csv, OmicsExpressionProteinCodingGenesTPMLogp1.csv
The expression file is optional; without it the expression stratification is
skipped and the module says so.

Record the release version. The manuscript must name it, and the reruns have to
use the same release as the submitted analysis or the numbers will not match.

Usage:
    python depmap_stratify.py --gene_effect CRISPRGeneEffect.csv \
        --cn OmicsCNGene.csv --expression OmicsExpression...csv --outdir ./results
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

DEPENDENCY_THRESHOLD = -0.5     # conventional DepMap cutoff for a dependency
STRONG_THRESHOLD = -1.0         # comparable to a common essential gene
MARGIN = 0.15                   # equivalence margin, Chronos units
HOM_MAX, HEM_MAX = 0.25, 0.75   # relative copy number, as in depmap_module.py

REFERENCE_GENES = ["RPL23A", "PSMA1", "POLR2A"]   # common essentials
NEGATIVE_CONTROLS = ["OR2W1", "CD19"]             # rarely essential


def find_col(df, gene):
    for c in df.columns:
        if re.match(rf"^{re.escape(gene)}\b", str(c)):
            return c
    return None


def classify_cn(v, hom_max=HOM_MAX, hem_max=HEM_MAX):
    if pd.isna(v):
        return np.nan
    if v < hom_max:
        return "HomDel"
    if v < hem_max:
        return "HemDel"
    return "Intact"


def tost(a, b, margin=MARGIN):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return np.nan, np.nan
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    if se == 0:
        return np.nan, float(diff)
    df = se ** 4 / ((a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
                    + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1))
    p = max(stats.t.sf((diff + margin) / se, df),
            stats.t.cdf((diff - margin) / se, df))
    return float(p), float(diff)


def _read_matrix(path, label):
    """Read a DepMap matrix keyed by cell line, whatever shape it ships in.

    The gene-effect file is indexed by ModelID directly. The WGS copy-number
    file instead carries ModelID as a column, alongside SequencingID and
    ModelConditionID, with one row per sequencing profile; the canonical row
    per model is flagged by IsDefaultEntryForModel. Reading it with
    index_col=0 picks up the row number and no join is possible.
    """
    df = pd.read_csv(path, low_memory=False)
    first = df.columns[0]
    if str(first).startswith("Unnamed") or df[first].astype(str).str.isnumeric().all():
        df = df.drop(columns=[first])
    key = next((c for c in ("ModelID", "DepMap_ID", "depmap_id")
                if c in df.columns), None)
    if key is None:
        # Already indexed by cell line, which is how the gene-effect file ships.
        df = pd.read_csv(path, index_col=0, low_memory=False)
        print(f"  {label}: indexed by {df.index.name or 'first column'}, "
              f"{len(df):,} lines", flush=True)
        return df
    flag = "IsDefaultEntryForModel"
    if flag in df.columns:
        n_before = len(df)
        df = df[df[flag].astype(str).str.lower().isin(("true", "1", "yes"))]
        print(f"  {label}: {n_before:,} rows reduced to {len(df):,} using "
              f"{flag}", flush=True)
    drop = [c for c in ("SequencingID", "ModelConditionID",
                        "IsDefaultEntryForMC", "IsDefaultEntryForModel",
                        "ProfileID", "Source", "WGSSequencingID")
            if c in df.columns]
    df = df.drop(columns=drop).drop_duplicates(key).set_index(key)
    print(f"  {label}: keyed on {key}, {len(df):,} lines", flush=True)
    return df


def run(gene_effect, cn, expression, outdir, gene="GLS"):
    os.makedirs(outdir, exist_ok=True)
    print("Loading DepMap files")
    ge = _read_matrix(gene_effect, "gene effect")
    cnd = _read_matrix(cn, "copy number")
    shared = ge.index.intersection(cnd.index)
    print(f"  {len(shared):,} cell lines in both files\n", flush=True)
    if len(shared) < 50:
        raise SystemExit(
            "Too few cell lines match between the two files. The identifiers "
            "differ, so they cannot be joined. Download OmicsCNGene.csv from "
            "the same DepMap release, which is keyed by ModelID.")

    gcol = find_col(ge, gene)
    pcol_cn = find_col(cnd, "PTEN")
    if gcol is None or pcol_cn is None:
        raise SystemExit(f"{gene} or PTEN not found in the DepMap files")

    df = pd.DataFrame({"gls": ge[gcol]})
    df["pten_cn"] = cnd[pcol_cn].reindex(df.index)
    df["tier"] = df["pten_cn"].apply(classify_cn)

    # ---- context: is GLS a dependency at all in this panel? --------------
    anchors = []
    for label, genes in (("common essential", REFERENCE_GENES),
                         ("negative control", NEGATIVE_CONTROLS)):
        for g in genes:
            c = find_col(ge, g)
            if c is None:
                continue
            v = ge[c].dropna()
            anchors.append({"gene": g, "class": label, "n": int(v.size),
                            "median_effect": round(float(v.median()), 4),
                            "pct_below_-0.5": round(100 * float((v < DEPENDENCY_THRESHOLD).mean()), 1)})
    v = df["gls"].dropna()
    anchors.append({"gene": gene, "class": "gene of interest", "n": int(v.size),
                    "median_effect": round(float(v.median()), 4),
                    "pct_below_-0.5": round(100 * float((v < DEPENDENCY_THRESHOLD).mean()), 1)})
    anchor_df = pd.DataFrame(anchors)
    anchor_df.to_csv(os.path.join(outdir, "depmap_dependency_context.csv"), index=False)

    print("Dependency context (is there anything for PTEN status to modify?)")
    print(anchor_df.to_string(index=False))
    pct = float(anchor_df.loc[anchor_df["gene"] == gene, "pct_below_-0.5"].iloc[0])
    print(f"\n  {gene} is a dependency (effect below {DEPENDENCY_THRESHOLD}) in "
          f"{pct:.1f}% of lines.")
    if pct < 10:
        print("  That is a narrow base. The tier comparison constrains little, "
              "and the manuscript should say so rather than presenting the null "
              "as independent confirmation.")

    # ---- expression stratification --------------------------------------
    have_expr = False
    if expression and os.path.exists(expression):
        ex = _read_matrix(expression, "expression")
        ecol = find_col(ex, "PTEN")
        if ecol is not None:
            df["pten_expr"] = ex[ecol].reindex(df.index)
            q = df["pten_expr"].quantile([0.25, 0.75])
            df["expr_group"] = np.where(df["pten_expr"] <= q.iloc[0], "low",
                                 np.where(df["pten_expr"] >= q.iloc[1], "high", "mid"))
            have_expr = True
            print("\nPTEN expression available; low and high groups are the "
                  "outer quartiles.")
    if not have_expr:
        print("\nPTEN expression file not supplied or PTEN column absent; "
              "the expression stratification is skipped.")

    # ---- tier comparisons -------------------------------------------------
    rows = []
    strata = {"all lines": df.index}
    if have_expr:
        strata["PTEN expression low"] = df.index[df["expr_group"] == "low"]
        strata["PTEN expression high"] = df.index[df["expr_group"] == "high"]

    for sname, idx in strata.items():
        sub = df.loc[idx].dropna(subset=["gls", "tier"])
        groups = {t: sub.loc[sub["tier"] == t, "gls"].values
                  for t in ("Intact", "HemDel", "HomDel")}
        row = {"stratum": sname,
               **{f"n_{t}": int(groups[t].size) for t in groups},
               **{f"median_{t}": (round(float(np.median(groups[t])), 4)
                                  if groups[t].size else None) for t in groups}}
        usable = [g for g in groups.values() if g.size >= 5]
        if len(usable) >= 2:
            row["kruskal_p"] = float(stats.kruskal(*usable).pvalue)
            sev = np.concatenate([np.full(groups[t].size, s)
                                  for t, s in (("Intact", 0), ("HemDel", 1), ("HomDel", 2))])
            vals = np.concatenate([groups[t] for t in ("Intact", "HemDel", "HomDel")])
            rho, p = stats.spearmanr(sev, vals)
            row["spearman_rho"] = round(float(rho), 4)
            row["spearman_p"] = float(p)
        if groups["Intact"].size >= 5 and groups["HomDel"].size >= 5:
            ptost, diff = tost(groups["HomDel"], groups["Intact"])
            row["diff_hom_vs_intact"] = round(diff, 4) if np.isfinite(diff) else None
            row["p_tost"] = None if np.isnan(ptost) else float(ptost)
            row["equivalence_margin"] = MARGIN
        # how many lines in this stratum are actually GLS-dependent
        row["pct_dependent"] = round(
            100 * float((sub["gls"] < DEPENDENCY_THRESHOLD).mean()), 1) if len(sub) else None
        rows.append(row)

    res = pd.DataFrame(rows)
    out = os.path.join(outdir, "depmap_stratified.csv")
    res.to_csv(out, index=False)
    print(f"\n[depmap_stratify] {len(res)} strata -> {out}")
    print(res.to_string(index=False))
    print("\n  Report the dependency context alongside the tier comparison. A "
          "null difference between tiers is only informative if the gene is a "
          "dependency in enough lines for a difference to be possible.")
    return res, anchor_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_effect", required=True)
    ap.add_argument("--cn", required=True)
    ap.add_argument("--expression", default=None)
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--gene", default="GLS")
    args = ap.parse_args()
    run(args.gene_effect, args.cn, args.expression, args.outdir, args.gene)


if __name__ == "__main__":
    main()
