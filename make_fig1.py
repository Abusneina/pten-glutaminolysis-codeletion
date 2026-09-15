#!/usr/bin/env python3
"""
make_fig1.py

Generates the replacement Figure 1 for PONE-D-26-35226.

Two problems with the submitted version are fixed.

1. Dimensions. The submitted file is 3000 pixels wide against a PLOS maximum of
   2250 at 300 dpi, so it is scaled to 75 percent at layout and the text shrinks
   with it. That is why the editor asked for a new copy. This version is drawn
   at 2250 x 1200, RGB, LZW compressed, no alpha channel, Arial at 8 to 12
   point, with no title inside the image.

2. Content. The submitted version carries a panel headed "What the human data
   show", stating the study's conclusions inside an Introduction schematic.
   Reviewer 2 objected, correctly: a figure introducing the question should
   pose it, not answer it. That panel is removed. What replaces it is the
   discriminating prediction each of the two competing explanations makes,
   which is what the reader needs before the Results.

The chromosomal strip is new. Without it, a reader meeting the co-deletion
hypothesis has no way to see why proximity matters. Positions are indicative
and drawn to relative scale within the interval; the figure legend should say
so rather than implying a physical map.

Usage:
    python make_fig1.py --outdir ./figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

# Non-blue palette, consistent across the revision's figures.
BURGUNDY = "#7B2D26"
AMBER = "#C08A2E"
TEAL = "#2E7D6B"
INK = "#1A1A1A"
GREY = "#6E6E6E"
FAINT = "#D8D2D0"
PANEL = "#F7F4F2"


def _box(ax, x, y, w, h, text, fc=PANEL, ec=FAINT, fs=9, weight="normal",
         color=INK, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, linewidth=lw,
                                 shrinkA=2, shrinkB=2))


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    # 7.5 x 4.0 inches at 300 dpi = 2250 x 1200 px, the PLOS maximum width.
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def head(letter, x, text):
        ax.text(x, 0.955, letter, fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + 0.026, 0.955, text, fontsize=9.5, fontweight="bold",
                color=BURGUNDY, va="center")

    # ---------------- Panel A: the signaling rationale -------------------
    head("A", 0.012, "Signaling rationale")

    col_x, box_w, box_h = 0.030, 0.185, 0.078
    ys = [0.800, 0.668, 0.536, 0.404]
    for y, lab in zip(ys, ["PTEN", "PI3K", "AKT", "mTORC1"]):
        _box(ax, col_x, y, box_w, box_h, lab, fs=9.5, weight="bold")
    _arrow(ax, col_x + box_w / 2, ys[0], col_x + box_w / 2, ys[1] + box_h,
           color=BURGUNDY, style="|-|", lw=1.3)
    ax.text(col_x + box_w + 0.008, (ys[0] + ys[1] + box_h) / 2,
            "inhibits", fontsize=7.8, color=BURGUNDY, va="center")
    for i in (1, 2):
        _arrow(ax, col_x + box_w / 2, ys[i], col_x + box_w / 2, ys[i + 1] + box_h)
    _arrow(ax, col_x + box_w / 2, ys[3], col_x + box_w / 2, 0.300)
    ax.text(col_x + box_w / 2 - 0.012, 0.352, "via MYC and\ntransporters",
            fontsize=7.8, color=GREY, va="center", ha="right")

    _box(ax, col_x - 0.018, 0.150, box_w + 0.036, 0.150,
         "Glutaminolysis program\nSLC1A5 \u00b7 GLS \u00b7 GLUD1\nGOT1 \u00b7 GPT2",
         fc="#F2EAE8", ec=BURGUNDY, fs=8.4)
    ax.text(col_x + box_w / 2, 0.112, "PTEN loss releases this axis",
            fontsize=7.6, color=GREY, ha="center", style="italic")

    # ---------------- Panel B: the two explanations ----------------------
    head("B", 0.300, "Two explanations")

    bx, bw = 0.318, 0.318
    _box(ax, bx, 0.632, bw, 0.248,
         "Regulatory\nLoss of PTEN restraint raises\nglutaminolytic transcription.\n"
         "Prediction: the signature rises\nIntact < HemDel < HomDel.",
         fc=PANEL, ec=TEAL, fs=8.2)
    ax.add_patch(Rectangle((bx, 0.632), 0.006, 0.248, facecolor=TEAL, edgecolor="none"))

    _box(ax, bx, 0.352, bw, 0.248,
         "Structural\nGLUD1 and GOT1 flank PTEN\nand are lost with it.\n"
         "Prediction: chr10 genes fall,\nothers do not.",
         fc=PANEL, ec=AMBER, fs=8.2)
    ax.add_patch(Rectangle((bx, 0.352), 0.006, 0.248, facecolor=AMBER, edgecolor="none"))

    _box(ax, bx, 0.122, bw, 0.198,
         "Discriminating tests\nDecompose by chromosome.\n"
         "Regress each gene on its own\ncopy number. Compare mutant\n"
         "copy-neutral with wild-type.",
         fc="#FBF9F8", ec=FAINT, fs=8.0)

    # ---------------- Panel C: the 10q interval --------------------------
    head("C", 0.668, "The 10q23-q24 interval")

    lx, lw_ = 0.700, 0.275
    y_line = 0.600
    ax.plot([lx, lx + lw_], [y_line, y_line], color=GREY, lw=2.2,
            solid_capstyle="round", zorder=2)

    # Indicative positions, relative scale within the interval.
    for frac, name, color, band, above in [
            (0.06, "GLUD1", BURGUNDY, "10q23.2", True),
            (0.145, "PTEN", INK, "10q23.31", False),
            (0.90, "GOT1", BURGUNDY, "10q24.2", True)]:
        x = lx + frac * lw_
        ax.plot([x, x], [y_line - 0.026, y_line + 0.026], color=color, lw=2.0, zorder=3)
        if above:
            ax.text(x, y_line + 0.042, name, fontsize=8.4, color=color,
                    ha="center", va="bottom", fontweight="bold")
            ax.text(x, y_line + 0.100, band, fontsize=7.2, color=GREY, ha="center",
                    va="bottom")
        else:
            ax.text(x, y_line - 0.052, name, fontsize=8.4, color=color,
                    ha="center", va="top", fontweight="bold")
            ax.text(x, y_line - 0.104, band, fontsize=7.2, color=GREY, ha="center",
                    va="top")

    ax.annotate("", xy=(lx + 0.06 * lw_, y_line - 0.175),
                xytext=(lx + 0.145 * lw_, y_line - 0.175),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(lx + 0.103 * lw_, y_line - 0.215, "~0.8 Mb", fontsize=7.2,
            color=GREY, ha="center", va="top")
    ax.annotate("", xy=(lx + 0.145 * lw_, y_line - 0.262),
                xytext=(lx + 0.90 * lw_, y_line - 0.262),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(lx + 0.52 * lw_, y_line - 0.300, "~11 Mb", fontsize=7.2,
            color=GREY, ha="center", va="top")

    _box(ax, lx - 0.004, 0.122, lw_ + 0.008, 0.135,
         "PTEN deletion commonly removes\nthis interval. Two of the five\n"
         "signature genes lie inside it;\nthe other three do not.",
         fc="#FBF9F8", ec=FAINT, fs=8.0)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    png = os.path.join(outdir, "Fig1.png")
    tif = os.path.join(outdir, "Fig1.tif")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    with Image.open(png) as im:
        im.convert("RGB").save(tif, format="TIFF", compression="tiff_lzw",
                               dpi=(300, 300))
    with Image.open(tif) as im:
        w, h = im.size
        print(f"Fig1.tif  {w} x {h} px, {im.mode}, LZW, 300 dpi")
        print(f"  PLOS width limit 2250: {'OK' if w <= 2250 else 'OVER'}")
        print(f"  PLOS height limit 2625: {'OK' if h <= 2625 else 'OVER'}")
    print(f"  {tif}\n  {png} (preview only, do not submit)")


LEGEND = """Fig 1. Competing explanations for an association between PTEN loss and the \
glutaminolysis signature.
(A) PTEN restrains PI3K/AKT/mTORC1 signaling, which raises glutaminolytic gene \
expression through MYC and transporter induction; on this account PTEN loss should \
increase the signature. (B) The regulatory and structural explanations make opposite \
predictions, and three tests distinguish them. (C) Two of the five signature genes, \
GLUD1 and GOT1, lie within the chromosome 10q23-q24 interval commonly removed by PTEN \
deletion; the remaining three lie on other chromosomes. Positions are indicative and \
drawn to relative scale within the interval rather than as a physical map."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./figures")
    args = ap.parse_args()
    build(args.outdir)
    with open(os.path.join(args.outdir, "Fig1_legend.txt"), "w") as fh:
        fh.write(LEGEND + "\n")
    print("\nLegend written to Fig1_legend.txt. It goes in the manuscript text, "
          "immediately after the paragraph where Fig 1 is first cited, not in "
          "the figure file.")


if __name__ == "__main__":
    main()
