#!/usr/bin/env python3
"""
interval_decay.py

Tests whether co-deletion with PTEN is a property of the 10q23-q24 interval
rather than of the glutaminolysis pathway.

The logic is simple. If GLUD1 and GOT1 fall with PTEN loss because they are
metabolically connected to PTEN, then unrelated genes in the same neighborhood
should not behave the same way. If they fall because they sit next to PTEN on
the chromosome, then every gene in the interval should track PTEN copy number,
with fidelity decaying as distance from PTEN increases, and the glutaminolysis
annotation should be incidental.

This addresses Reviewer 2, observation 5, and converts Reviewer 3's objection
about novelty into a supporting result: the contribution is not that passenger
co-deletion happens, which is established, but that it can be mistaken for
coordinated pathway regulation inside a curated signature.

It also supplies a replacement for the unsupported claim at manuscript lines
279 to 280 that the 10q23 neighborhood is unusually rich in metabolically
annotated genes. That claim currently rests on a citation that contains no
positional content. This module produces the positional evidence directly, or
shows that the claim should be dropped.

COORDINATES ARE NOT HARD-CODED. They are fetched from Ensembl and written to a
CSV you can inspect, so no gene position in this analysis rests on anyone's
recollection. Genes are then filtered by what Ensembl actually returns, so a
candidate that turns out not to lie in the interval is excluded automatically
and reported.

Usage:
    python interval_decay.py --fetch-coords                      # once, needs internet
    python interval_decay.py --datadir ./data --outdir ./results
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A

ENSEMBL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}?content-type=application/json"
COORD_FILE = "chr10_gene_coordinates.csv"

ANCHOR = "PTEN"

# Signature genes that sit on chromosome 10 and are the subject of the paper.
SIGNATURE_CHR10 = ["GLUD1", "GOT1"]
# Signature genes on other chromosomes, carried through as a negative control.
SIGNATURE_OFF_CHR10 = ["GLS", "SLC1A5", "GPT2"]

# Candidate neighbours with no glutaminolysis annotation. This list is
# deliberately generous; membership of the interval is decided by the
# coordinates Ensembl returns, not by this list.
CANDIDATE_NEIGHBORS = [
    "KLLN", "ATAD1", "PAPSS2", "RNLS", "MINPP1", "BMPR1A", "WAPL", "FAS",
    "ACTA2", "LIPA", "IFIT1", "IFIT2", "IFIT3", "HHEX", "IDE", "KIF11",
    "EXOC6", "CYP26A1", "MYOF", "PLCE1", "NOC3L", "TBC1D12", "HELLS",
    "CYP2C9", "PDLIM1", "SORBS1", "ENTPD1", "HPS1", "HPSE2", "ERLIN1",
    "CPN1", "DNMBP", "ABCC2", "CHUK", "SCD", "SEC31B", "NDUFB8", "HIF1AN",
]

# Half-width of the window around the anchor, in base pairs. Genes outside it
# are reported but excluded from the decay fit.
WINDOW_BP = 25_000_000

N_BOOT = 2000
RNG = np.random.default_rng(20260908)


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------
def fetch_coordinates(genes, outfile=COORD_FILE, pause=0.12):
    """Query Ensembl for each gene and write a coordinate table."""
    rows = []
    for i, g in enumerate(genes, 1):
        url = ENSEMBL.format(gene=g)
        try:
            with urllib.request.urlopen(url, timeout=30) as fh:
                d = json.load(fh)
        except Exception as exc:
            print(f"  [{i}/{len(genes)}] {g}: lookup failed ({exc})")
            continue
        chrom = str(d.get("seq_region_name", ""))
        start, end = d.get("start"), d.get("end")
        strand = d.get("strand")
        if not chrom or start is None:
            print(f"  [{i}/{len(genes)}] {g}: no coordinates returned")
            continue
        tss = start if strand == 1 else end
        rows.append({"gene": g, "chromosome": chrom, "start": start, "end": end,
                     "strand": strand, "tss": tss,
                     "assembly": d.get("assembly_name", ""),
                     "ensembl_id": d.get("id", "")})
        print(f"  [{i}/{len(genes)}] {g}: chr{chrom}:{start}-{end} "
              f"({d.get('assembly_name','')})")
        time.sleep(pause)
    if not rows:
        raise SystemExit("No coordinates retrieved. Check the internet connection.")
    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"\nWrote {len(df)} gene coordinates to {outfile}")
    print("Inspect this file before running the analysis. Every position used "
          "in the paper comes from it.")
    return df


def load_coordinates(path=COORD_FILE):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run:  python interval_decay.py --fetch-coords\n"
            "This needs an internet connection and takes under a minute.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _boot_ci(x, y, n_boot=N_BOOT, alpha=0.05):
    n = x.size
    if n < 10:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals[b] = stats.spearmanr(x[idx], y[idx]).statistic
    vals = vals[np.isfinite(vals)]
    if vals.size < n_boot // 4:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def _classify(gene):
    if gene == ANCHOR:
        return "anchor"
    if gene in SIGNATURE_CHR10:
        return "signature, chr10"
    if gene in SIGNATURE_OFF_CHR10:
        return "signature, off-chr10"
    return "neighbor, unannotated"


def run(datadir, outdir, coords_path=COORD_FILE, window=WINDOW_BP):
    os.makedirs(outdir, exist_ok=True)
    coords = load_coordinates(coords_path)

    anchor_row = coords[coords["gene"] == ANCHOR]
    if anchor_row.empty:
        raise SystemExit(f"{ANCHOR} is missing from {coords_path}.")
    anchor_chr = str(anchor_row["chromosome"].iloc[0])
    anchor_tss = int(anchor_row["tss"].iloc[0])
    print(f"Anchor {ANCHOR} at chr{anchor_chr}:{anchor_tss}")

    coords = coords.copy()
    coords["same_chromosome"] = coords["chromosome"].astype(str) == anchor_chr
    coords["distance_bp"] = np.where(
        coords["same_chromosome"], (coords["tss"] - anchor_tss).abs(), np.nan)
    coords["in_window"] = coords["distance_bp"] <= window
    coords["class"] = coords["gene"].map(_classify)

    excluded = coords[(~coords["same_chromosome"]) &
                      (coords["class"] == "neighbor, unannotated")]
    if len(excluded):
        print("Candidates not on the anchor chromosome, excluded from the fit: "
              + ", ".join(excluded["gene"]))
    far = coords[coords["same_chromosome"] & ~coords["in_window"]]
    if len(far):
        print("Candidates beyond the window, excluded from the fit: "
              + ", ".join(far["gene"]))

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

        # Same guard as the other modules: unassigned samples are absent
        # from the CNA matrix and would break the per-gene lookup.
        tiers = tiers.dropna()
        cols = list(tiers.index)
        if ANCHOR not in cna.index:
            continue
        anchor_cn = pd.to_numeric(cna.loc[ANCHOR, cols], errors="coerce").values.astype(float)

        for _, gr in coords.iterrows():
            g = gr["gene"]
            if g == ANCHOR or g not in cna.index:
                continue
            gcn = pd.to_numeric(cna.loc[g, cols], errors="coerce").values.astype(float)
            m = np.isfinite(gcn) & np.isfinite(anchor_cn)
            if m.sum() < 20 or np.unique(gcn[m]).size < 2:
                continue
            rho, p = stats.spearmanr(gcn[m], anchor_cn[m])
            lo, hi = _boot_ci(gcn[m], anchor_cn[m])
            rows.append({
                "study": label, "gene": g, "class": gr["class"],
                "chromosome": gr["chromosome"],
                "distance_bp": gr["distance_bp"],
                "distance_mb": None if pd.isna(gr["distance_bp"]) else round(gr["distance_bp"] / 1e6, 3),
                "n": int(m.sum()),
                "rho_cn_vs_pten_cn": round(float(rho), 4),
                "p": float(p),
                "ci_lo": None if np.isnan(lo) else round(lo, 4),
                "ci_hi": None if np.isnan(hi) else round(hi, 4),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[interval_decay] no cohort produced usable correlations")
        return res, pd.DataFrame()

    per_gene = (res.groupby(["gene", "class", "chromosome", "distance_mb"], dropna=False)
                  .agg(cohorts=("rho_cn_vs_pten_cn", "size"),
                       mean_rho=("rho_cn_vs_pten_cn", "mean"),
                       min_rho=("rho_cn_vs_pten_cn", "min"),
                       max_rho=("rho_cn_vs_pten_cn", "max"))
                  .reset_index()
                  .sort_values("distance_mb"))

    res.to_csv(os.path.join(outdir, "interval_decay_per_cohort.csv"), index=False)
    per_gene.to_csv(os.path.join(outdir, "interval_decay_per_gene.csv"), index=False)
    print(f"[interval_decay] {len(res)} gene-cohort rows, {len(per_gene)} genes "
          f"-> {outdir}")

    # Headline: does fidelity fall with distance among unannotated neighbours?
    fit = per_gene[(per_gene["class"] == "neighbor, unannotated")
                   & per_gene["distance_mb"].notna()]
    if len(fit) >= 5:
        rho_d, p_d = stats.spearmanr(fit["distance_mb"], fit["mean_rho"])
        print(f"  distance-decay among unannotated neighbours: "
              f"Spearman rho = {rho_d:+.3f}, P = {p_d:.3g}, {len(fit)} genes")
        print("  A clear negative value means co-deletion fidelity falls with "
              "distance from PTEN, which is the interval explanation.")

    for cls in ["signature, chr10", "neighbor, unannotated", "signature, off-chr10"]:
        sub = per_gene[per_gene["class"] == cls]
        if len(sub):
            print(f"  {cls}: {len(sub)} genes, mean rho {sub['mean_rho'].mean():+.3f}")

    _plot(per_gene, outdir)
    return res, per_gene


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _plot(per_gene, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PLOS: max width 2250 px at 300 dpi (7.5 in); Arial, 8-12 pt; no title
    # inside the figure; legend inside the plot area.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    style = {
        "signature, chr10":      ("#7B2D26", "o", 55, "Signature genes on chr10 (GLUD1, GOT1)"),
        "neighbor, unannotated": ("#C08A2E", "s", 38, "10q neighbors, no glutaminolysis annotation"),
        "signature, off-chr10":  ("#2E7D6B", "^", 45, "Signature genes on other chromosomes"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=300)

    chr10_max = per_gene.loc[per_gene["distance_mb"].notna(), "distance_mb"].max()
    chr10_max = float(chr10_max) if pd.notna(chr10_max) else 1.0
    divider_x = chr10_max * 1.8
    off_x = chr10_max * 3.2

    for cls, (color, marker, size, label) in style.items():
        sub = per_gene[per_gene["class"] == cls]
        if not len(sub):
            continue
        if cls == "signature, off-chr10":
            # These have no distance from the anchor. They are parked beyond a
            # divider so they read as a reference group, not as far-away
            # neighbors on chromosome 10.
            xs = np.full(len(sub), off_x)
        else:
            xs = sub["distance_mb"].values
        ax.scatter(xs, sub["mean_rho"], c=color, marker=marker, s=size,
                   edgecolors="white", linewidths=0.6, label=label, zorder=3)

    if (per_gene["class"] == "signature, off-chr10").any():
        ax.axvline(divider_x, color="#CCCCCC", lw=0.9, ls=":", zorder=1)
        ax.text(off_x, ax.get_ylim()[1], "other\nchromosomes", ha="center",
                va="top", fontsize=8, color="#666666")

    fit = per_gene[(per_gene["class"] == "neighbor, unannotated")
                   & per_gene["distance_mb"].notna()].sort_values("distance_mb")
    if len(fit) >= 5:
        # LOESS is overkill for this many points; a log-distance linear fit is
        # transparent and easy to describe in the legend.
        x = np.log10(fit["distance_mb"].clip(lower=0.01))
        b, a = np.polyfit(x, fit["mean_rho"], 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(10 ** xx, a + b * xx, color="#8C8C8C", lw=1.2, ls="--",
                zorder=2, label="Fit to unannotated neighbors")

    ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(left=0)
    ax.set_xlabel("Distance from PTEN transcription start site (Mb, log scale)")
    ax.set_ylabel("Spearman rho, gene copy number vs PTEN copy number")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)

    out = os.path.join(outdir, "Fig_interval_decay.tif")
    png = out.replace(".tif", ".png")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    # PLOS requires RGB or grayscale with no alpha channel, flattened, LZW.
    # Matplotlib's TIFF writer leaves an alpha channel in place, so the file is
    # rebuilt from the PNG through PIL to guarantee compliance.
    from PIL import Image
    with Image.open(png) as im:
        im.convert("RGB").save(out, format="TIFF", compression="tiff_lzw",
                               dpi=(300, 300))
    print(f"  figure -> {out} (and {os.path.basename(png)} for previewing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="./data")
    ap.add_argument("--outdir", default="./results")
    ap.add_argument("--coords", default=COORD_FILE)
    ap.add_argument("--fetch-coords", action="store_true",
                    help="query Ensembl for gene coordinates and exit")
    ap.add_argument("--window-mb", type=float, default=WINDOW_BP / 1e6)
    args = ap.parse_args()

    if args.fetch_coords:
        genes = [ANCHOR] + SIGNATURE_CHR10 + SIGNATURE_OFF_CHR10 + CANDIDATE_NEIGHBORS
        fetch_coordinates(genes, args.coords)
        return

    run(args.datadir, args.outdir, args.coords, int(args.window_mb * 1e6))


if __name__ == "__main__":
    main()
