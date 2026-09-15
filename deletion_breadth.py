#!/usr/bin/env python3
"""
deletion_breadth.py

Answers Reviewer 2, observation 6: whether thresholded GISTIC calls are
capturing broad 10q loss rather than focal PTEN deletion, and whether the
reported deep-deletion frequencies are therefore inflated.

Comparing our frequencies against a published figure would settle little, since
published frequencies differ by call type, threshold and cohort version. The
reviewer's underlying question is answerable directly with the data in hand.

Three tests.

  1. Frequency. PTEN deep-deletion frequency per cohort, so the numbers are on
     the record and can be checked against any external source.

  2. Breadth. For each tumor called deep-deleted at PTEN, how many of the
     surrounding 10q23-q24 genes are also called deep-deleted, using the
     coordinates already fetched from Ensembl. A genuinely focal deletion
     removes PTEN and its nearest neighbours; a broad arm-level event removes
     genes across the whole interval. The distribution of breadth tells you
     which kind of event the thresholded call is picking up, and the span in
     megabases makes it concrete.

  3. Purity. Whether deep-deletion calls track tumor purity. Low purity masks
     homozygous deletion, so if anything this biases frequencies downward, in
     the opposite direction to the reviewer's concern. Reporting it closes the
     question in both directions rather than only one.

Requires chr10_gene_coordinates.csv, written by interval_decay.py --fetch-coords,
and tumor purity merged by merge_purity.py. Both are optional; the module reports
what it could not do rather than failing.

Usage:
    python deletion_breadth.py --datadir .\\data --outdir .\\results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A

COORD_FILE = "chr10_gene_coordinates.csv"
ANCHOR = "PTEN"
# Genes within this distance of PTEN count as the local neighbourhood; beyond
# it, involvement indicates a broad event rather than a focal one.
FOCAL_MB = 2.0


def load_coords(path):
    if not os.path.exists(path):
        return None
    c = pd.read_csv(path)
    anchor = c[c.gene == ANCHOR]
    if anchor.empty:
        return None
    chrom = str(anchor.chromosome.iloc[0])
    tss = int(anchor.tss.iloc[0])
    c = c[c.chromosome.astype(str) == chrom].copy()
    c["distance_mb"] = (c.tss - tss).abs() / 1e6
    return c.sort_values("distance_mb")


def _purity(study_dir, cols):
    path = os.path.join(study_dir, "data_clinical_sample.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = [str(c).strip().upper().replace(" ", "_").lstrip("#")
                  for c in df.columns]
    if "SAMPLE_ID" not in df.columns or "TUMOR_PURITY" not in df.columns:
        return None
    df["SAMPLE_ID"] = df["SAMPLE_ID"].astype(str).str[:15]
    s = df.drop_duplicates("SAMPLE_ID").set_index("SAMPLE_ID")["TUMOR_PURITY"]
    v = pd.to_numeric(s, errors="coerce").reindex(cols)
    return v if v.notna().sum() >= 20 else None


def run(datadir, outdir, coords_path=COORD_FILE):
    os.makedirs(outdir, exist_ok=True)
    coords = load_coords(coords_path)
    if coords is None:
        print(f"{coords_path} not found; breadth test will be skipped.")
        print("Run: python interval_decay.py --fetch-coords\n")

    rows, breadth_rows = [], []
    dirs = [d for d in sorted(os.listdir(datadir))
            if os.path.isdir(os.path.join(datadir, d))]

    for i, code in enumerate(dirs, 1):
        sdir = os.path.join(datadir, code)
        label = A.STUDY_LABELS.get(code, code)
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
        print(f"  [{i}/{len(dirs)}] {label}", flush=True)

        tiers = tiers.dropna()
        cols = list(tiers.index)
        n = len(cols)
        n_hom = int((tiers == "HomDel").sum())
        n_hem = int((tiers == "HemDel").sum())

        row = {"study": label, "n_tumors": n,
               "n_homdel": n_hom, "pct_homdel": round(100 * n_hom / n, 2),
               "n_hemdel": n_hem, "pct_hemdel": round(100 * n_hem / n, 2)}

        # ---- breadth ------------------------------------------------------
        if coords is not None:
            present = [g for g in coords.gene if g in cna.index and g != ANCHOR]
            if present:
                sub = cna.loc[present, cols].apply(pd.to_numeric, errors="coerce")
                dist = coords.set_index("gene").loc[present, "distance_mb"]
                hom_samples = [c for c in cols if tiers[c] == "HomDel"]
                per_sample = []
                for s in hom_samples:
                    v = sub[s]
                    deep = v[v <= -2]
                    if not len(deep):
                        per_sample.append((0, 0.0, 0))
                        continue
                    d = dist.loc[deep.index]
                    per_sample.append((len(deep), float(d.max()),
                                       int((d > FOCAL_MB).sum())))
                if per_sample:
                    ngenes = np.array([p[0] for p in per_sample])
                    span = np.array([p[1] for p in per_sample])
                    beyond = np.array([p[2] for p in per_sample])
                    row.update({
                        "interval_genes_tested": len(present),
                        "median_neighbors_also_deep": int(np.median(ngenes)),
                        "median_span_mb": round(float(np.median(span)), 2),
                        "pct_focal_under_2mb": round(
                            100 * float((span <= FOCAL_MB).mean()), 1),
                        "pct_broad_over_2mb": round(
                            100 * float((span > FOCAL_MB).mean()), 1),
                    })
                    for s, (g, sp, b) in zip(hom_samples, per_sample):
                        breadth_rows.append({"study": label, "sample": s,
                                             "neighbors_deep": g,
                                             "span_mb": round(sp, 3),
                                             "beyond_2mb": b})

        # ---- purity -------------------------------------------------------
        pur = _purity(sdir, cols)
        if pur is not None:
            is_hom = (tiers == "HomDel").values
            a = pur.values[is_hom]
            b = pur.values[~is_hom]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) >= 5 and len(b) >= 5:
                row["purity_homdel"] = round(float(np.median(a)), 3)
                row["purity_other"] = round(float(np.median(b)), 3)
                row["purity_p"] = float(stats.mannwhitneyu(a, b).pvalue)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        print("No cohort produced results.")
        return res

    res.to_csv(os.path.join(outdir, "deletion_breadth_by_cohort.csv"), index=False)
    if breadth_rows:
        pd.DataFrame(breadth_rows).to_csv(
            os.path.join(outdir, "deletion_breadth_per_sample.csv"), index=False)

    print(f"\n[deletion_breadth] {len(res)} cohorts -> {outdir}\n")
    show = [c for c in ["study", "n_tumors", "pct_homdel", "pct_hemdel",
                        "median_neighbors_also_deep", "median_span_mb",
                        "pct_focal_under_2mb"] if c in res.columns]
    print(res[show].to_string(index=False))

    if "pct_focal_under_2mb" in res.columns:
        m = res.pct_focal_under_2mb.mean()
        print(f"\n  Across cohorts, a median {res.median_span_mb.median():.2f} Mb "
              f"of the interval is co-deleted in PTEN deep-deleted tumors, and "
              f"{m:.0f} percent of such tumors have deletions confined within "
              f"{FOCAL_MB:.0f} Mb of PTEN.")
        print("  A high focal percentage means the thresholded call is picking "
              "up focal events, not arm-level loss, which is the reviewer's "
              "concern answered directly.")
    if "purity_p" in res.columns:
        lower = int((res.purity_homdel < res.purity_other).sum())
        print(f"\n  Tumor purity is lower in deep-deleted tumors in {lower} of "
              f"{int(res.purity_homdel.notna().sum())} cohorts. Low purity masks "
              f"homozygous deletion, so this biases frequencies downward rather "
              f"than upward.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--coords", default=COORD_FILE)
    args = ap.parse_args()
    run(args.datadir, args.outdir, args.coords)


if __name__ == "__main__":
    main()
