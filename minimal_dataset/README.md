# Minimal data set for PONE-D-26-35226

Generated 2026-09-13 by make_minimal_dataset.py.

## What this is

One row per tumor, carrying every variable from which the figures and tables in
the manuscript are computed. Deposited to satisfy the PLOS ONE requirement for a
minimal data set sufficient to replicate the reported findings.

## Provenance

Derived from TCGA PanCancer Atlas data obtained through the cBioPortal datahub.
Primary data are third-party and remain available from cBioPortal; this file is
the analysis-ready layer between those sources and the manuscript. No new data
were generated. TCGA barcodes are de-identified by design and no clinical
narrative is included.

## Files

- minimal_dataset_tumor_level.csv: 6186 tumors, 14 cohorts
- data_dictionary.csv: column definitions
- excluded_studies.csv: every study assessed but not analyzed, with the reason
- MANIFEST.csv: file sizes and SHA-256 checksums

## Cohorts included

- BLCA: 404 tumors
- BRCA: 1068 tumors
- CESC: 290 tumors
- COADREAD: 590 tumors
- GBM: 148 tumors
- HNSC: 509 tumors
- LIHC: 361 tumors
- LUSC: 484 tumors
- OV: 295 tumors
- PRAD: 488 tumors
- SARC: 251 tumors
- SKCM: 367 tumors
- STAD: 410 tumors
- UCEC: 521 tumors

## Reproducing the manuscript from this file

The analysis code at the accompanying repository regenerates every table and
figure. Copy-number status is the thresholded GISTIC2 call; expression is
log2(RSEM + 1) z-scored within each cohort across all samples of that cohort,
not against a diploid reference subset.
