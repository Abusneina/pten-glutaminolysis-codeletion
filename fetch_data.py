#!/usr/bin/env python3
"""
fetch_data.py
Download TCGA PanCancer Atlas studies from the cBioPortal datahub for the
PTEN copy-number dosage / glutaminolysis study.

By default this fetches ALL 32 solid-tumor PanCancer Atlas studies, enabling
the pan-cancer design with a pre-specified inclusion rule applied downstream
by analyze.py (a study enters the analysis only if it has enough tumors in
each PTEN dosage tier). Use --primary to fetch only the four initial-design
primary-analysis studies (UCEC, PRAD, BRCA, GBM), or --studies to name a set.

The datahub stores each file on GitHub using git-LFS. This script resolves
each file through the GitHub LFS batch API (host: github.com) and downloads
the object from GitHub's LFS storage host. It works on any normal machine,
and inside a restricted sandbox only if the egress allowlist includes
github.com AND github-cloud.githubusercontent.com (+ media.githubusercontent.com).

No cBioPortal account or API key is required; all PanCancer Atlas studies
are public. Study identifiers follow the pattern <code>_tcga_pan_can_atlas_2018.

Usage:
    python3 fetch_data.py --outdir ./data            # all qualifying tumor types
    python3 fetch_data.py --primary --outdir ./data  # the four primary studies
"""
import argparse
import json
import os
import subprocess
import urllib.request

REPO = "cBioPortal/datahub"
BRANCH = "master"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/public/{study}/{fname}"
BATCH = "https://github.com/{repo}.git/info/lfs/objects/batch"

# All solid-tumor TCGA PanCancer Atlas studies (one per cancer type).
# Hematologic (LAML) and the redundant combined studies are omitted; DLBC is
# retained as a solid-tissue lymphoma. Study code -> human-readable label.
STUDIES = {
    "acc_tcga_pan_can_atlas_2018":  "Adrenocortical Carcinoma",
    "blca_tcga_pan_can_atlas_2018": "Bladder Urothelial Carcinoma",
    "brca_tcga_pan_can_atlas_2018": "Breast Invasive Carcinoma",
    "cesc_tcga_pan_can_atlas_2018": "Cervical Squamous Cell Carcinoma",
    "chol_tcga_pan_can_atlas_2018": "Cholangiocarcinoma",
    "coadread_tcga_pan_can_atlas_2018": "Colorectal Adenocarcinoma",
    "dlbc_tcga_pan_can_atlas_2018": "Diffuse Large B-cell Lymphoma",
    "esca_tcga_pan_can_atlas_2018": "Esophageal Adenocarcinoma",
    "gbm_tcga_pan_can_atlas_2018":  "Glioblastoma Multiforme",
    "hnsc_tcga_pan_can_atlas_2018": "Head and Neck Squamous Cell Carcinoma",
    "kich_tcga_pan_can_atlas_2018": "Kidney Chromophobe",
    "kirc_tcga_pan_can_atlas_2018": "Kidney Renal Clear Cell Carcinoma",
    "kirp_tcga_pan_can_atlas_2018": "Kidney Renal Papillary Cell Carcinoma",
    "lgg_tcga_pan_can_atlas_2018":  "Brain Lower Grade Glioma",
    "lihc_tcga_pan_can_atlas_2018": "Liver Hepatocellular Carcinoma",
    "luad_tcga_pan_can_atlas_2018": "Lung Adenocarcinoma",
    "lusc_tcga_pan_can_atlas_2018": "Lung Squamous Cell Carcinoma",
    "meso_tcga_pan_can_atlas_2018": "Mesothelioma",
    "ov_tcga_pan_can_atlas_2018":   "Ovarian Serous Cystadenocarcinoma",
    "paad_tcga_pan_can_atlas_2018": "Pancreatic Adenocarcinoma",
    "pcpg_tcga_pan_can_atlas_2018": "Pheochromocytoma and Paraganglioma",
    "prad_tcga_pan_can_atlas_2018": "Prostate Adenocarcinoma",
    "sarc_tcga_pan_can_atlas_2018": "Sarcoma",
    "skcm_tcga_pan_can_atlas_2018": "Skin Cutaneous Melanoma",
    "stad_tcga_pan_can_atlas_2018": "Stomach Adenocarcinoma",
    "tgct_tcga_pan_can_atlas_2018": "Testicular Germ Cell Tumors",
    "thca_tcga_pan_can_atlas_2018": "Thyroid Carcinoma",
    "thym_tcga_pan_can_atlas_2018": "Thymoma",
    "ucec_tcga_pan_can_atlas_2018": "Uterine Corpus Endometrial Carcinoma",
    "ucs_tcga_pan_can_atlas_2018":  "Uterine Carcinosarcoma",
    "uvm_tcga_pan_can_atlas_2018":  "Uveal Melanoma",
}

PRIMARY = [
    "ucec_tcga_pan_can_atlas_2018",
    "prad_tcga_pan_can_atlas_2018",
    "brca_tcga_pan_can_atlas_2018",
    "gbm_tcga_pan_can_atlas_2018",
]

FILES = [
    "data_mrna_seq_v2_rsem.txt",
    "data_mutations.txt",
    "data_cna.txt",
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
]


def _read_pointer(study, fname):
    url = RAW.format(repo=REPO, branch=BRANCH, study=study, fname=fname)
    try:
        txt = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    except Exception as e:
        return None, None, f"pointer fetch failed: {e}"
    oid = size = None
    for line in txt.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split("sha256:")[1].strip()
        elif line.startswith("size "):
            size = int(line.split()[1])
    if oid is None:
        return "INLINE", txt, None
    return oid, size, None


def _resolve_lfs(oid, size):
    body = json.dumps({
        "operation": "download", "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode()
    req = urllib.request.Request(
        BATCH.format(repo=REPO), data=body,
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-type": "application/vnd.git-lfs+json"})
    resp = json.load(urllib.request.urlopen(req, timeout=60))
    return resp["objects"][0]["actions"]["download"]["href"]


def _download(href, dest):
    subprocess.run(["curl", "-sL", "-o", dest, href], check=True)


def fetch_study(study, outdir):
    sdir = os.path.join(outdir, study)
    os.makedirs(sdir, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(sdir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  [skip] {study}/{fname}")
            continue
        oid, size, err = _read_pointer(study, fname)
        if err:
            print(f"  [WARN] {study}/{fname}: {err}")
            continue
        if oid == "INLINE":
            with open(dest, "w") as fh:
                fh.write(size)
            print(f"  [ok]   {study}/{fname} (inline)")
            continue
        try:
            href = _resolve_lfs(oid, size)
            _download(href, dest)
            got = os.path.getsize(dest)
            flag = "ok" if got == size else "SIZE-MISMATCH"
            print(f"  [{flag}] {study}/{fname} ({got:,} bytes)")
        except Exception as e:
            print(f"  [FAIL] {study}/{fname}: {e}")
            print("         In a sandbox, add github-cloud.githubusercontent.com to egress.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data")
    ap.add_argument("--primary", action="store_true",
                    help="fetch only the four primary-analysis studies")
    ap.add_argument("--studies", nargs="*", default=None,
                    help="explicit list of study IDs to fetch")
    args = ap.parse_args()
    if args.studies:
        targets = args.studies
    elif args.primary:
        targets = PRIMARY
    else:
        targets = list(STUDIES)
    print(f"[fetch] {len(targets)} studies -> {args.outdir}")
    for study in targets:
        print(f"[study] {study} -- {STUDIES.get(study, '?')}")
        fetch_study(study, args.outdir)


if __name__ == "__main__":
    main()
