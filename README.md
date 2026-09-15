# pten-glutaminolysis-codeletion

Analysis code for "Passenger co-deletion confounds glutaminolysis signatures
anchored on PTEN loss: a cautionary case for location-aware signature design"
(PLOS ONE, PONE-D-26-35226).

No new data were generated. All primary data are public: TCGA PanCancer Atlas
expression, copy-number, mutation and clinical data from the cBioPortal datahub,
consensus tumor purity from the TCGA PanCanAtlas ABSOLUTE tables, and CRISPR
gene-effect and copy-number data from DepMap Public 26Q1.

The minimal data set required to replicate every reported finding is deposited
at figshare, DOI 10.6084/m9.figshare.33716476. This repository is archived at
Zenodo, DOI 10.5281/zenodo.21195209.

## Requirements

```
pip install -r requirements.txt
```

pandas, numpy, scipy, statsmodels, matplotlib, lifelines, openpyxl.

## Obtaining the data

```
python fetch_data.py --outdir ./data
python fetch_mutations_api.py --datadir ./data
python check_data.py --datadir ./data
python merge_purity.py --datadir ./data --purity-file TCGA_mastercalls.abs_tables_JSedit.fixed.txt
```

`fetch_data.py` downloads 31 solid-tumor studies from the cBioPortal datahub.
Mutation files are served through Git LFS and may fail on bandwidth limits;
`fetch_mutations_api.py` retrieves PTEN mutations through the cBioPortal REST
API instead, which is all the analysis reads. `check_data.py` detects truncated
downloads, which are otherwise reused silently. The ABSOLUTE purity table is
available from the GDC PanCanAtlas publications page.

## Running the analysis

Run the modules in the order below. Each writes its own CSV to `./results`, and
the figure scripts read that folder.

| Script | Produces |
|---|---|
| `fetch_data.py` | Downloads the 31 solid-tumor studies from the cBioPortal datahub |
| `analyze.py` | Dose-response, signature scoring, survival |
| `chr10_test.py` | Chromosome-10 versus off-chromosome-10 sub-signatures (Table 2) |
| `aneuploidy_partial.py` | Partial correlations, aneuploidy and gene-level (S3 Table) |
| `mutation_experiment.py` | Mutation natural experiment with equivalence testing (S4 Table) |
| `interval_decay.py` | Co-deletion across 10q23-q24 by distance (Fig 4, S6 Table) |
| `deletion_breadth.py` | Whether deletions are focal or arm-level |
| `effect_sizes.py` | Effect sizes with bootstrap intervals, monotonicity |
| `mediation_myc.py` | MYC mediation, transcript and Hallmark target scores |
| `msi_stratification.py` | Mutation analysis stratified by MSI and mutational burden |
| `prad_adjust.py` | Prostate covariate model; multiple-testing across cohorts |
| `depmap_stratify.py` | GLS dependency by PTEN dosage, with dependency context |
| `make_minimal_dataset.py` | The deposited minimal data set |
| `make_fig1.py`, `make_figures.py` | All figures, to PLOS specification |
| `make_synthetic.py` | Synthetic fixtures for validation only |

`interval_decay.py --fetch-coords` retrieves gene coordinates from Ensembl and
writes them to a CSV, so no genomic position in the analysis is hard-coded.

## Cohort selection

A study enters the analysis only if it has at least ten tumors in each of the
intact, hemizygous and homozygous deletion tiers. Applied to all 31 solid-tumor
studies, fourteen qualify. The excluded studies and their tier counts are listed
in S5 Table.

Which cohorts appear in which figure is decided in `make_figures.py`, not in
`analyze.py`. `fig2_signature_by_dosage` ranks the qualifying studies by the
median difference between the homozygous and intact tiers and draws the six
largest declines as Fig 2; the remaining eight go to S2 Fig. `SURVIVAL_COHORTS`
fixes the four survival panels as UCEC, GBM, PRAD and BRCA, the cohorts the
Results paragraph discusses. Both are display choices; every qualifying study is
analyzed identically and all fourteen appear in Fig 3, Fig 5 and Fig 6.

## Figures

```
python make_fig1.py --outdir ./figures
python -u make_figures.py --datadir ./data --resultsdir ./results --outdir ./figures
```

`make_fig1.py` needs no data and also writes `Fig1_legend.txt`, which belongs in
the manuscript text rather than in the image. `make_figures.py` prints the cohort
list for Fig 2 and S2 Fig, and for Fig 3 it prints how significance was
determined and names the cohorts that fail.

Significance in Fig 3 follows the Methods: Benjamini-Hochberg across the
included tumor types. If `effect_sizes.csv` carries an adjusted column the
figure reads it; `effect_sizes.py` as deposited does not write one, so the
correction is computed in `make_figures.py` from the raw Spearman P values.
Either route must give twelve of fourteen significant, with GBM and STAD the
exceptions, which is what the Padj column of Table 1 reports. If the printed
count differs, the figure and the table disagree and the run should not be
used.

Both scripts write a TIFF for submission and a PNG for previewing, at most
2250 x 2625 pixels, 300 dpi, RGB, LZW compressed, with no title inside the image
and legends inside the plot area.

## Validation

`make_synthetic.py` generates fixtures with linked copy number, mutation calls
and aneuploidy scores, so each module can be exercised end to end without the
real data. The fixtures are synthetic and must never be reported as results.
