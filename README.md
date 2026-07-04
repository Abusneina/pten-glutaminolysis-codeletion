# PTEN Copy-Number Loss and Glutaminolytic Gene Expression: Analysis Code

Reproducible analysis code for the manuscript:

**Chromosomal Co-Deletion, Not Signaling, Explains the Association Between PTEN
Copy-Number Loss and Reduced Glutaminolytic Gene Expression: A Pan-Cancer
Analysis.** Abusneina AM, Elwirfli SM. University of Benghazi, Libya.

This code reproduces every table and figure in the manuscript from publicly
available data. No new data were generated.

## Data sources (public, no permissions required)
- TCGA PanCancer Atlas: cBioPortal datahub — https://github.com/cBioPortal/datahub
- DepMap Public (CRISPRGeneEffect, OmicsCNGene, Model): https://depmap.org/portal/download

## Requirements
    pip install -r requirements.txt

## Scripts
- `fetch_data.py`     Download TCGA PanCancer Atlas studies from the cBioPortal datahub.
- `analyze.py`        Aim 1 (dose-response) and survival, with the pre-specified
                      inclusion rule and three-tier PTEN dosage classification.
- `myc_analysis.py`   Tests MYC as a candidate transcriptional mediator (negative control).
- `chr10_test.py`     Tests chromosome-10 co-deletion of GLUD1/GOT1 as the mechanism.
- `depmap_module.py`  GLS CRISPR dependency by PTEN dosage (DepMap).
- `make_synthetic.py` Synthetic fixtures for pipeline validation only (not used for results).

## Reproduce the analysis
    # 1. Obtain TCGA data (all qualifying tumor types)
    python3 fetch_data.py --outdir ./data
    # 2. Aim 1 + survival
    python3 analyze.py --datadir ./data --outdir ./results
    # 3. MYC mechanism test
    python3 myc_analysis.py --datadir ./data --outdir ./results
    # 4. Chromosome-10 co-deletion test
    python3 chr10_test.py --datadir ./data --outdir ./results
    # 5. DepMap dependency (after downloading DepMap CSVs)
    python3 depmap_module.py --gene_effect CRISPRGeneEffect.csv \
        --cn OmicsCNGene.csv --model Model.csv --outdir ./results

## Methods summary
PTEN tiers from GISTIC calls (-2 HomDel, -1 HemDel, >=0 Intact). Glutaminolysis
signature = mean per-gene log2 z-score of GLS, SLC1A5, GOT1, GLUD1, GPT2.
Dose-response by Spearman correlation with loss severity, Benjamini-Hochberg
corrected across included tumor types. Inclusion rule: >=10 tumors in each of
HemDel, HomDel, and Intact. DepMap copy number is linear relative copy number
(neutral ~1.0); verify units for your release.

## License
MIT (see LICENSE).

## Citation
If you use this code, please cite the manuscript above. A permanent archived
version of this repository is available at Zenodo (DOI assigned on deposit).
