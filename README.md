# REZCUE

Analysis code for the REZCUE: 3D spatial multi-omics profiling from archived FFPE tissue

| Folder | Language | Purpose |
|---|---|---|
| `segmentation/` | MATLAB | 2D U-Net++ multi-class segmentation of 3D REZCUE image stacks |
| `alignment/` | MATLAB | Registration of CosMx slices to the 3D REZCUE volume |
| `cosmx/` | Python, R | CosMx spatial transcriptomics and mass-spectrometry analysis and figure generation |
| `env/` | – | Conda environment files |

---

## 1. System requirements

### Software

| Component | Version used | Notes |
|---|---|---|
| MATLAB | R2022b | Image Processing, Deep Learning, Computer Vision and Parallel Computing toolboxes |
| Python env `cell2loc_env` | 3.12.8 | For `brain_cosmx.py`. Pinned in `env/cell2loc_env.yml` |
| Python env `cosmx` | 3.14.7 | For `CA2_MassSpec.py`. Pinned in `env/cosmx_env.yml` |
| R | 4.4.2 | For `CA2_cosmx.R`. Seurat 4.0, dplyr, ggplot2, ggsci, tidyr, tibble, reshape2 |

### Operating systems tested
- MATLAB scripts: Windows R2022b
- Python/R scripts: macOS (Apple M1, x86-64)

### Hardware
- The Python and R analyses run on a standard desktop computer.
- U-Net++ training needs an NVIDIA GPU with CUDA support (tested on **TODO GPU, VRAM**). It falls back to the CPU if no GPU is found, but training is then much slower.
- Registering whole-slide CosMx data to 3D volumes needs **TODO GB** RAM.
- `CA2_MassSpec.py` queries the UniProt REST API, so it needs internet access.

---

## 2. Installation

```bash
git clone https://github.com/heiminglai/REZCUE.git
cd REZCUE
```

**Python** (conda):
```bash
# for brain_cosmx.py
conda env create -f env/cell2loc_env.yml
conda activate cell2loc_env
pip install squidpy==TODO

# for CA2_MassSpec.py
conda env create -f env/cosmx_env.yml
conda activate cosmx
pip install scikit-learn requests
```
Both files pin builds for macOS x86-64. On Linux or Windows, install the key packages listed above into environments with the same Python versions instead.

**R**:
```r
install.packages(c("Seurat", "dplyr", "ggplot2", "ggsci", "tidyr", "tibble", "reshape2"))
```

Typical install time on a normal desktop computer: 30 min** (mostly creating the conda environments).

---

## 3. Demo

Demo data: **TODO: link to demo dataset (e.g. Zenodo DOI)**. Download it and place the files in `cosmx/`.

```bash
cd cosmx
conda activate cosmx
python CA2_MassSpec.py            # input: protein.normalized.intensity.xls
Rscript CA2_cosmx.R               # input: seuratObject_CA2_scaled.RDS
```

**Expected output**
- `CA2_MassSpec.py` writes to `figures/`: `01_pca_regions.png`, three gradient volcano plots, and one venn diagram.
- `CA2_cosmx.R` writes to `./CosMx/CA/figure/` (create this folder first): marker dot plot, UMAP, spatial maps of cell types, clones and niches, cell-type proportion plots, and divergent-gene plots. These correspond to Fig. 5F–J and Supplementary Fig. 6.

Expected run time on a normal desktop computer: 5 min for `CA2_MassSpec.py`, 25min for `CA2_cosmx.R`.

---

## 4. Instructions for use

Set the input and output paths at the top of each script, then run it. The MATLAB scripts are meant to be run section by section.

**Segmentation** (`segmentation/`)
- `AreaSegmentation_pilot_unetpp.m`: trains a 2D U-Net++ on an annotated multi-channel stack, then segments every slice of a raw 3D stack into tissue compartments.
  - Inputs: a raw interleaved-channel TIFF stack, plus training images and masks. Set them with `masterfolder`, `rawStackPath`, `annImgStackPath` and `annMaskStackPath`.
  - Set `numChannels`, `classes` and `labelIDs` to match your data. Network depth, epochs and augmentation are set in the same block.
  - Outputs go to `trainingoutput/EncoderDepth<N>/run_<timestamp>/`: the trained model, per-slice predictions, the training curve and a run metadata file.
  - The random seed is fixed with `rng(42)`.

**Alignment** (`alignment/`)
- `aged_globalmorphology_crop_resize.m`: crops each tissue slice out of the whole-slide CosMx morphology and cell-segmentation images and downsizes them 8× in X and Y. These smaller images are used for manual pre-alignment. Edit the crop `indices` for your sample.
- `postmanualRegistration.m`: registers the pre-aligned CosMx slices to the 3D REZCUE volume. It applies a rigid 3D transform from landmark points (`static_pt`), then demons non-rigid registration. It writes the transformed XYZ coordinate grids and cell masks for each slice (`slice<N>_gridXYZ_tformed.tif`, `slice<N>_CosMxSegCellMask_tformed.tif`). Set the image paths, landmarks and `sliceNum` for your sample.
- `buildForwardFromPositionMapScaled.m`: function that turns a warped position map into a forward mapping, from full-resolution CosMx coordinates to positions in the registered output image.
- `tiffwriteVolume.m`: helper that writes 3–5D arrays to multi-page TIFF, switching to BigTIFF automatically for files over 4 GB.

**CosMx and proteomics** (`cosmx/`)

- `brain_cosmx.py` (`cell2loc_env`): QC, Harmony integration, Leiden clustering and annotation of brain CosMx data. Input is an `.h5ad` file, set with `pth`; output folder is set with `output_dir`.
- `CA2_cosmx.R`: Seurat clustering, cell-type, clone and niche analysis. Input is a Seurat `.RDS` object; output folder is set with `fig_path`.
- `CA2_MassSpec.py` (`cosmx`): regional proteomics (PCA, gradient trajectories, Venn). Input is a tab-separated table of normalized protein intensities, set with `DATA_PATH`; output folder is set with `OUTDIR`.

---


The processed data is available via Dryad (DOI: 10.5061/dryad.2ngf1vj56) and the raw data may be made available upon request to the corresponding author (H.M.L).  

---


