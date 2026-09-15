#!/usr/bin/env python3
"""
make_synthetic.py  --  VALIDATION ONLY.

Generates cBioPortal-format fixtures with a three-tier PTEN dosage structure.
A monotonic glutaminolysis effect is injected in most types; a few are given
very small HemDel/HomDel counts so the inclusion rule in analyze.py can be
exercised (they should be excluded).

Extended relative to the v1.0 release so that the aneuploidy and mutation
modules can be validated end to end. The v1.0 outputs are unchanged; three
things are added:

  1. Linked copy number for the chromosome-10 signature genes. GLUD1 and GOT1
     now track PTEN copy number with distance-dependent fidelity (GLUD1 is
     ~0.8 Mb from PTEN, GOT1 ~11 Mb), while the off-chromosome-10 genes stay
     independent. In v1.0 every gene except PTEN had copy number fixed at 0,
     which made the co-deletion mechanism untestable on fixtures.
  2. An ANEUPLOIDY_SCORE column in data_clinical_sample.txt, correlated with
     loss severity, so partial correlation can be exercised. One study is
     given no aneuploidy column at all, reproducing the LUSC case.
  3. A data_mutations.txt file carrying PTEN variants of each class, including
     a copy-neutral inactivating-mutation group, so the natural experiment can
     be exercised.

These data are synthetic and must never be reported as results.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)

EXTRA = ["TP53", "MYC", "AKT1", "EGFR", "KRAS", "BRCA1", "RB1", "CDKN2A"]
SIG = ["GLS", "SLC1A5", "GOT1", "GLUD1", "GPT2"]

# Chromosome-10 signature genes and the fidelity with which their copy number
# follows PTEN. Nearer genes are co-deleted more reliably.
CHR10_FIDELITY = {"GLUD1": 0.92, "GOT1": 0.70}

# Additional 10q23-q24 genes carrying no glutaminolysis annotation, used by the
# interval analysis. Fidelity again falls with distance from PTEN.
INTERVAL_GENES = {"KLLN": 0.97, "ATAD1": 0.90, "PAPSS2": 0.78, "RNLS": 0.62}

VARIANT_CLASSES = [
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "Silent",
]
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"}


def _linked_cna(pten_tier, fidelity, n):
    """Copy number for a gene co-deleted with PTEN at the given fidelity."""
    follows = rng.random(n) < fidelity
    independent = rng.choice([0, 0, 0, -1], size=n)
    return np.where(follows, pten_tier, independent)


def make_study(path, n, frac_hem, frac_hom, step,
               mut_rate=0.08, hypermutated=False, aneuploidy=True):
    os.makedirs(path, exist_ok=True)
    samples = [f"TCGA-XX-{i:04d}-01" for i in range(n)]
    patients = [s[:12] for s in samples]

    r = rng.random(n)
    tier = np.where(r < frac_hom, -2, np.where(r < frac_hom + frac_hem, -1, 0))
    sev = np.where(tier == -2, 2, np.where(tier == -1, 1, 0))

    # ---- mutation status -------------------------------------------------
    # Inactivating PTEN mutations are drawn independently of copy number, so a
    # copy-neutral mutant group exists. Hypermutated studies get many more,
    # reproducing the endometrial case.
    rate = mut_rate * (4.0 if hypermutated else 1.0)
    mutated = rng.random(n) < rate
    vclass = np.where(
        mutated,
        rng.choice(VARIANT_CLASSES, size=n, p=[.34, .18, .16, .08, .12, .06, .06]),
        "",
    )
    inactivating = mutated & (vclass != "Silent")

    genes = ["PTEN"] + SIG + EXTRA + list(INTERVAL_GENES)
    expr = pd.DataFrame(index=genes, columns=samples, dtype=float)
    for g in genes:
        expr.loc[g] = rng.lognormal(6, 0.6, n)

    # PTEN transcript falls with copy loss; mutation does not change transcript
    # level, which is the whole point of the natural experiment.
    expr.loc["PTEN", samples] = rng.lognormal(6.5 - 0.9 * sev, 0.4)

    # ---- copy number, generated before expression so the chromosome-10 genes
    # can be driven by their own dosage rather than by PTEN severity ---------
    cna = pd.DataFrame(0, index=genes, columns=samples)
    cna.loc["PTEN", samples] = tier
    for g, fid in CHR10_FIDELITY.items():
        cna.loc[g, samples] = _linked_cna(tier, fid, n)
    for g, fid in INTERVAL_GENES.items():
        cna.loc[g, samples] = _linked_cna(tier, fid, n)
    for g in ["GLS", "SLC1A5", "GPT2"]:
        cna.loc[g, samples] = rng.choice([0, 0, 0, 0, -1, 1], size=n)

    # Signature gene expression. This is the fixture's positive control for the
    # whole study: the chromosome-10 genes respond to their OWN copy number,
    # not to PTEN, so conditioning on that copy number should abolish their
    # apparent association with PTEN loss. The off-chromosome-10 genes carry no
    # PTEN effect at all, so conditioning should leave them unchanged.
    for g in CHR10_FIDELITY:
        own_sev = -cna.loc[g, samples].values.astype(float)   # 0, 1, 2
        expr.loc[g, samples] = expr.loc[g, samples].astype(float) * (2 ** (-step * own_sev))
    for g in ["GLS", "SLC1A5", "GPT2"]:
        pass  # no PTEN-linked effect injected

    e = expr.reset_index().rename(columns={"index": "Hugo_Symbol"})
    e.insert(1, "Entrez_Gene_Id", range(1, len(e) + 1))
    e.to_csv(os.path.join(path, "data_mrna_seq_v2_rsem.txt"), sep="\t", index=False)

    # ---- write copy number -----------------------------------------------
    c = cna.reset_index().rename(columns={"index": "Hugo_Symbol"})
    c.insert(1, "Entrez_Gene_Id", range(1, len(c) + 1))
    c.to_csv(os.path.join(path, "data_cna.txt"), sep="\t", index=False)

    # ---- survival --------------------------------------------------------
    base = rng.exponential(60, n)
    time = (base * np.where(sev == 2, 0.5, np.where(sev == 1, 0.75, 1.0))).clip(0.1, 240)
    event = rng.random(n) < np.where(sev == 2, 0.6, np.where(sev == 1, 0.45, 0.32))
    with open(os.path.join(path, "data_clinical_patient.txt"), "w") as fh:
        fh.write("#Patient Identifier\tOverall Survival (Months)\tOverall Survival Status\n")
        fh.write("#Patient Identifier\tOS Months\tOS Status\n")
        fh.write("#STRING\tNUMBER\tSTRING\n")
        fh.write("#1\t1\t1\n")
        fh.write("PATIENT_ID\tOS_MONTHS\tOS_STATUS\n")
        for p, tt, ev in zip(patients, time, event):
            fh.write(f"{p}\t{round(tt,1)}\t{'1:DECEASED' if ev else '0:LIVING'}\n")

    # ---- clinical sample, with aneuploidy score --------------------------
    sample_tbl = {"SAMPLE_ID": samples, "PATIENT_ID": patients}
    if aneuploidy:
        # Correlated with severity but far from collinear, so partialling it out
        # should leave a genuine co-deletion signal intact.
        sample_tbl["ANEUPLOIDY_SCORE"] = np.clip(
            rng.normal(8 + 3.0 * sev, 5.0, n).round(0), 0, 39
        ).astype(int)
    pd.DataFrame(sample_tbl).to_csv(
        os.path.join(path, "data_clinical_sample.txt"), sep="\t", index=False)

    # ---- mutations -------------------------------------------------------
    rows = []
    for s, m, vc in zip(samples, mutated, vclass):
        if not m:
            continue
        rows.append({
            "Hugo_Symbol": "PTEN",
            "Entrez_Gene_Id": 5728,
            "Tumor_Sample_Barcode": s,
            "Variant_Classification": vc,
            "Variant_Type": "SNP" if "Del" not in vc and "Ins" not in vc else "INS",
            "HGVSp_Short": "p.X" + str(rng.integers(1, 404)) + "X",
        })
    # A little non-PTEN noise so gene filtering is actually exercised.
    for s in rng.choice(samples, size=max(5, n // 20), replace=False):
        rows.append({
            "Hugo_Symbol": "TP53", "Entrez_Gene_Id": 7157,
            "Tumor_Sample_Barcode": s, "Variant_Classification": "Missense_Mutation",
            "Variant_Type": "SNP", "HGVSp_Short": "p.R175H",
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(path, "data_mutations.txt"), sep="\t", index=False)

    return int(inactivating.sum())


if __name__ == "__main__":
    root = os.environ.get("SYNTH_ROOT", "./synthetic_data")
    # (code, n, frac_hem, frac_hom, effect step, hypermutated, aneuploidy)
    spec = [
        ("ucec_tcga_pan_can_atlas_2018", 420, 0.28, 0.10, 0.55, True,  True),
        ("prad_tcga_pan_can_atlas_2018", 380, 0.16, 0.16, 0.42, False, True),
        ("brca_tcga_pan_can_atlas_2018", 440, 0.22, 0.07, 0.32, False, True),
        ("gbm_tcga_pan_can_atlas_2018",  270, 0.30, 0.12, 0.50, False, True),
        ("blca_tcga_pan_can_atlas_2018", 300, 0.32, 0.09, 0.40, False, True),
        ("skcm_tcga_pan_can_atlas_2018", 300, 0.40, 0.10, 0.35, False, True),
        ("sarc_tcga_pan_can_atlas_2018", 260, 0.34, 0.11, 0.30, False, True),
        ("stad_tcga_pan_can_atlas_2018", 290, 0.22, 0.08, 0.25, False, True),
        # No aneuploidy column: reproduces the LUSC case.
        ("lusc_tcga_pan_can_atlas_2018", 380, 0.35, 0.12, 0.20, False, False),
        # Should FAIL the inclusion rule (HomDel < 10).
        ("thca_tcga_pan_can_atlas_2018", 200, 0.02, 0.01, 0.10, False, True),
        ("kirp_tcga_pan_can_atlas_2018", 150, 0.03, 0.01, 0.10, False, True),
    ]
    for code, n, fhem, fhom, step, hyper, aneu in spec:
        k = make_study(os.path.join(root, code), n, fhem, fhom, step,
                       hypermutated=hyper, aneuploidy=aneu)
        print(f"  {code}: n={n}, inactivating PTEN mutations={k}")
    print(f"synthetic data ({len(spec)} tumor types) written to", root)
