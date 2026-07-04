#!/usr/bin/env python3
"""
make_synthetic.py  --  VALIDATION ONLY.
Generates cBioPortal-format fixtures for a BROADER set of tumor types with a
three-tier PTEN dosage structure. A monotonic glutaminolysis effect is injected
in most types; a few are given very small HemDel/HomDel counts so the
inclusion rule in analyze.py can be exercised (they should be excluded).
These data are synthetic and must never be reported as results.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
EXTRA = ["TP53", "MYC", "AKT1", "EGFR", "KRAS", "BRCA1", "RB1", "CDKN2A"]
SIG = ["GLS", "SLC1A5", "GOT1", "GLUD1", "GPT2"]


def make_study(path, n, frac_hem, frac_hom, step):
    os.makedirs(path, exist_ok=True)
    samples = [f"TCGA-XX-{i:04d}-01" for i in range(n)]
    patients = [s[:12] for s in samples]
    r = rng.random(n)
    tier = np.where(r < frac_hom, -2, np.where(r < frac_hom + frac_hem, -1, 0))
    sev = np.where(tier == -2, 2, np.where(tier == -1, 1, 0))

    genes = ["PTEN"] + SIG + EXTRA
    expr = pd.DataFrame(index=genes, columns=samples, dtype=float)
    for g in genes:
        expr.loc[g] = rng.lognormal(6, 0.6, n)
    expr.loc["PTEN", samples] = rng.lognormal(6.5 - 0.9 * sev, 0.4)
    for g in SIG:
        expr.loc[g, samples] = expr.loc[g, samples].astype(float) * (2 ** (step * sev))

    e = expr.reset_index().rename(columns={"index": "Hugo_Symbol"})
    e.insert(1, "Entrez_Gene_Id", range(1, len(e) + 1))
    e.to_csv(os.path.join(path, "data_mrna_seq_v2_rsem.txt"), sep="\t", index=False)

    cna = pd.DataFrame(0, index=genes, columns=samples)
    cna.loc["PTEN", samples] = tier
    c = cna.reset_index().rename(columns={"index": "Hugo_Symbol"})
    c.insert(1, "Entrez_Gene_Id", range(1, len(c) + 1))
    c.to_csv(os.path.join(path, "data_cna.txt"), sep="\t", index=False)

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
    pd.DataFrame({"SAMPLE_ID": samples, "PATIENT_ID": patients}).to_csv(
        os.path.join(path, "data_clinical_sample.txt"), sep="\t", index=False)


if __name__ == "__main__":
    root = "/home/claude/pipeline/synthetic_data"
    # (code, n, frac_hem, frac_hom, effect step). Last two have tiny loss
    # fractions -> should FAIL the inclusion rule (HomDel < 10).
    spec = [
        ("ucec_tcga_pan_can_atlas_2018", 220, 0.28, 0.10, 0.55),  # primary
        ("prad_tcga_pan_can_atlas_2018", 180, 0.16, 0.16, 0.42),  # primary
        ("brca_tcga_pan_can_atlas_2018", 240, 0.22, 0.07, 0.32),  # primary
        ("gbm_tcga_pan_can_atlas_2018",  170, 0.30, 0.12, 0.50),  # primary
        ("blca_tcga_pan_can_atlas_2018", 200, 0.32, 0.09, 0.40),  # extension
        ("skcm_tcga_pan_can_atlas_2018", 200, 0.40, 0.10, 0.35),  # extension
        ("sarc_tcga_pan_can_atlas_2018", 160, 0.34, 0.11, 0.30),  # extension
        ("stad_tcga_pan_can_atlas_2018", 190, 0.22, 0.08, 0.25),  # extension
        ("thca_tcga_pan_can_atlas_2018", 200, 0.02, 0.01, 0.10),  # should be EXCLUDED
        ("kirp_tcga_pan_can_atlas_2018", 150, 0.03, 0.01, 0.10),  # should be EXCLUDED
    ]
    for code, n, fhem, fhom, step in spec:
        make_study(os.path.join(root, code), n, fhem, fhom, step)
    print(f"synthetic data ({len(spec)} tumor types) written to", root)
