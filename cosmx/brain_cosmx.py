#cell2loc_env
import scanpy as sc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
sc.settings.set_figure_params(dpi=100)
output_dir = "./REZCUE/CosMx/brain_5_6/figure/"

pth = './REZCUE/CosMx/brain_5_6/'
adata = sc.read_h5ad(pth + 'brain_spe_data.h5ad')
adata

adata.obs['tissue_name'] = adata.obs['tissue_id'].map({'S2': 'REZCUE', 'S1': 'Control', 'S3': 'Control'})

adata.var["NegPrb"] = adata.var_names.str.startswith("NegPrb")
sc.pp.calculate_qc_metrics(adata, qc_vars=["NegPrb"], inplace=True)

adata.obs["total_counts_NegPrb"].sum() / adata.obs["total_counts"].sum() * 100

adata.obs['fov'] = adata.obs['fov'].astype('str')

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].set_title("Total transcripts per cell")
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    ax=axs[0],
)

axs[1].set_title("Unique transcripts per cell")
sns.histplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    ax=axs[1],
)

axs[2].set_title("Transcripts per FOV")
sns.histplot(
    adata.obs.groupby("fov").sum(numeric_only=True)["total_counts"],
    kde=False,
    ax=axs[2],
)

import seaborn as sns
import matplotlib.pyplot as plt

adata_sub = adata[adata.obs['tissue_id'].isin(['S1', 'S2'])].copy()
#adata_sub.obs['tissue_name'] = adata_sub.obs['tissue_id'].map({'S1': 'REZCUE', 'S2': 'Control'})
adata_sub.obs['tissue_name'] = adata_sub.obs['tissue_id'].map({'S2': 'REZCUE', 'S1': 'Control'})

# Aggregate transcripts per FOV per tissue
df = (
    adata_sub.obs
    .groupby(["tissue_name", "fov"])["total_counts"]
    .sum()
    .reset_index()
)

# Plot overlapping histograms
plt.figure(figsize=(6, 4))

sns.histplot(
    data=df,
    x="total_counts",
    hue="tissue_name",
    bins=50,
    element="step",     # cleaner than filled bars
    stat="density",     # normalize for comparison
    common_norm=False,  # each tissue normalized independently
)

plt.xscale("log")  # important for count data
plt.xlabel("Transcripts per FOV")
plt.ylabel("Density")
plt.title("Transcripts per FOV across tissues")
plt.tight_layout()

# plot the distribution of transcripts over on the tissue
import seaborn as sns
# Clean publication style
sns.set_style("white")

df = pd.DataFrame(adata_sub.obsm['spatial'])
df['total_counts'] = adata_sub.obs['total_counts'].values
df.columns = ['x', 'y', 'total_counts']
df['tissue'] = adata_sub.obs['tissue_name'].values
df.head()

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="total_counts",
    color="tissue",
    s=1,
    alpha=0.8,
    linewidth=0,
    ax=ax
)

# Publication formatting
ax.set_aspect("equal")
ax.grid(False)
sns.despine(ax=ax)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xticks([])
ax.set_yticks([])

# Uncomment if coordinates come from microscopy images
# ax.invert_yaxis()

# Move legend outside
leg = ax.legend(
    title="Total counts",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=False,
    markerscale=1.2,      # enlarge legend dots
    scatterpoints=1,
    fontsize=9,
    title_fontsize=10,
    ncol=1
)

plt.tight_layout()
plt.show()

sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=400)

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

adata_sub = adata[adata.obs['tissue_id'].isin(['S1', 'S2'])].copy()
adata_sub.obs['tissue_name'] = adata_sub.obs['tissue_id'].map({'S2': 'REZCUE', 'S1': 'Control'})

sc.pl.umap(
    adata,
    color=[
        "total_counts",
        "n_genes_by_counts",
        "leiden",
    ],
    wspace=0.4,
)

sc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon"
)

sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
import pandas as pd
markers = pd.DataFrame({
    cluster: adata.uns["rank_genes_groups"]["names"][cluster][:10]
    for cluster in adata.obs["leiden"].cat.categories
})

adata_sub = adata[adata.obs['tissue_id'].isin(['S1', 'S2'])].copy()
adata_sub

import harmonypy as hm
adata_sub= sc.read_h5ad(output_dir + "adata_sub.h5ad")
ho = hm.run_harmony(
    adata_sub.obsm["X_pca"],
    adata_sub.obs,
    "tissue_name"
)

adata_sub.obsm["X_pca_harmony"] = ho.Z_corr
sc.pp.neighbors(adata_sub, use_rep='X_pca_harmony')
sc.tl.umap(adata_sub)
sc.tl.leiden(adata_sub)
sc.settings.set_figure_params(dpi=300)
sc.pl.umap(
    adata_sub,
    color=[
        "tissue_name",
        "leiden",
    ],
    alpha=0.5,
    wspace=0.4
)

df = pd.DataFrame(adata_sub.obsm['spatial'])
df.columns = ['x', 'y']
df['leiden'] = adata_sub.obs['leiden'].values
df['condition'] = adata_sub.obs['tissue_name'].values



sc.settings.set_figure_params(dpi=100)
# fraction of each cell type
frac = (
    pd.crosstab(df["condition"], df["leiden"], normalize="index")
    .T
)

# order cell types if desired
cell_order = frac.index
colors = adata_sub.uns["leiden_colors"]

fig, ax = plt.subplots(figsize=(3,6))

bottom = [0,0]

for ct, color in zip(cell_order, colors):
    vals = frac.loc[ct]

    ax.bar(
        frac.columns,
        vals,
        bottom=bottom,
        color=color,
        width=0.7,
        edgecolor="none",
        label=ct
    )

    bottom += vals.values

ax.set_ylim(0,1)
ax.set_ylabel("Fraction of cells")
ax.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()


import seaborn as sns

# Clean publication style
sns.set_style("white")

fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="leiden",
    #palette=adata_sub.uns["leiden_colors"],
    s=1,
    alpha=0.8,
    linewidth=0,
    ax=ax
)

# Publication formatting
ax.set_aspect("equal")
ax.grid(False)
sns.despine(ax=ax)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])

# Uncomment if coordinates come from microscopy images
# ax.invert_yaxis()

# Move legend outside
leg = ax.legend(
    title="Cell type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=False,
    markerscale=1.2,      # enlarge legend dots
    scatterpoints=1,
    fontsize=9,
    title_fontsize=10,
    ncol=1
)

plt.tight_layout()
plt.show()

sc.tl.rank_genes_groups(
    adata_sub,
    groupby="leiden",
    method="wilcoxon"
)

sc.pl.rank_genes_groups(adata_sub, n_genes=20, sharey=False)


import pandas as pd
markers = pd.DataFrame({
    cluster: adata_sub.uns["rank_genes_groups"]["names"][cluster][:50]
    for cluster in adata_sub.obs["leiden"].cat.categories
})
markers.shape


celltype_dict = {
    '0':  "GABAergic neuron",
    '1':  "Glutamatergic neuron",
    '2':  "Oligodendrocyte",
    '3':  "Astrocyte",
    '4':  "Glutamatergic neuron", # "Glutamatergic neuron (IT/Satb2+)"
    '5':  "GABAergic neuron (Pcp4+)",
    '6':  "Neuron (unresolved)",
    '7':  "VLMC", #	Astrocyte (Apod+)
    '8':  "Pvalb interneuron",
    '9':  "Oligodendrocyte",
    '10': "GABAergic neuron (Tac1+)",
    '11': "Thalamic excitatory neuron (Tcf7l2+)", #"Glutamatergic neuron (L6 corticothalamic-like)",
    '12': "Glutamatergic neuron",
    '13': "Microglia", #"Choroid plexus epithelial",
    '14': "Choroid plexus epithelial"
}

celltype_colors = {
    # Excitatory neurons (blue-green)
    "Glutamatergic neuron": "#25478F",
    "Thalamic excitatory neuron (Tcf7l2+)": "#1AE4E8",

    # Inhibitory neurons (red-orange)
    "GABAergic neuron": "#E64B35",
    "GABAergic neuron (Pcp4+)": "#F39B7F",
    "GABAergic neuron (Tac1+)": "#D55E00",
    "Pvalb interneuron": "#B2182B",
    # Glia
    "Oligodendrocyte": "#108C10",
    "Astrocyte": "#EE99EE",
    "Microglia": "#C306AD",
    "VLMC": "#2C2C25",

    # Other
    "Choroid plexus epithelial": "#F0E442",
    "Neuron (unresolved)": "#9E9E9E",
}

adata.uns["cell_type_colors"] = [
    celltype_colors[c]
    for c in adata_sub.obs["cell_type"].cat.categories
]

adata_sub.obs["cell_type"] = adata_sub.obs["leiden"].map(celltype_dict).astype("category")



celltype_colors = {
    # Excitatory neurons (blue-green)
    "Glutamatergic neuron": "#25478F",
    "Thalamic excitatory neuron (Tcf7l2+)": "#1AE4E8",

    # Inhibitory neurons (red-orange)
    "GABAergic neuron": "#E64B35",
    "GABAergic neuron (Pcp4+)": "#F39B7F",
    "GABAergic neuron (Tac1+)": "#D55E00",
    "Pvalb interneuron": "#B2182B",
    # Glia
    "Oligodendrocyte": "#108C10",
    "Astrocyte": "#EE99EE",
    "Microglia": "#C306AD",
    "VLMC": "#2C2C25",

    # Other
    "Choroid plexus epithelial": "#F0E442",
    "Neuron (unresolved)": "#9E9E9E",
}

adata_sub.uns["cell_type_colors"] = [
    celltype_colors[c]
    for c in adata_sub.obs["cell_type"].cat.categories
]

sc.settings.set_figure_params(dpi=300)
sc.pl.umap(
    adata_sub,
    color=[
        "tissue_name",
        "cell_type",
    ],
    alpha=0.5,
    wspace=0.4
)


df['celltype'] = adata_sub.obs['cell_type'].values
df['condition'] = adata_sub.obs['tissue_name'].values
# fraction of each cell type
frac = (
    pd.crosstab(df["condition"], df["celltype"], normalize="index")
    .T
)

# order cell types if desired
cell_order = frac.index
colors = adata_sub.uns["cell_type_colors"]

fig, ax = plt.subplots(figsize=(3,6))

bottom = [0,0]

for ct, color in zip(cell_order, colors):
    vals = frac.loc[ct]

    ax.bar(
        frac.columns,
        vals,
        bottom=bottom,
        color=color,
        width=0.7,
        edgecolor="none",
        label=ct
    )

    bottom += vals.values

ax.set_ylim(0,1)
ax.set_ylabel("Fraction of cells")
ax.legend(
    bbox_to_anchor=(1.02,1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="celltype",
    palette=adata_sub.uns["cell_type_colors"],
    s=1,
    alpha=0.8,
    linewidth=0,
    ax=ax
)

# Publication formatting
ax.set_aspect("equal")
ax.grid(False)
sns.despine(ax=ax)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xticks([])
ax.set_yticks([])

# Uncomment if coordinates come from microscopy images
# ax.invert_yaxis()

# Move legend outside
leg = ax.legend(
    title="Cell type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
    frameon=False,
    markerscale=1.2,      # enlarge legend dots
    scatterpoints=1,
    fontsize=9,
    title_fontsize=10,
    ncol=1
)

plt.tight_layout()
plt.show()

summary = (
    pd.crosstab(df["celltype"], df["condition"], normalize="columns")
)

summary["difference"] = (
    summary["REZCUE"] - summary["Control"]
)

# red = upregulated in REZCUE (higher fraction than Control), blue = downregulated
colors = np.where(summary["difference"] > 0, "#c0392b", "#2a52b2")

fig, ax = plt.subplots(figsize=(8, 4))

y = range(len(summary))

ax.axvline(0, color="grey", ls="--")
ax.scatter(summary["difference"], y, s=60, color=colors, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(summary.index)
ax.set_xlabel("Difference in fraction (REZCUE − Control)")

xmax = summary["difference"].abs().max() * 1.3
ax.set_xlim(-xmax, xmax)

plt.tight_layout()


# intersect the marker genes with the genes in the adata_sub object
marker_panel = {
    "Glutamatergic neuron": ["Slc17a7", "Camk2a", "Nrgn", "Satb2"],
    "GABAergic neuron": ["Gad1", "Gad2", "Slc32a1", "Pcp4"],
    "Pvalb interneuron": ["Pvalb", "Gad1", "Kcnc1", "Igfbpl1"],
    "Oligodendrocyte": ["Mbp", "Plp1", "Mobp", "Mag"],
    "Astrocyte": ["Apoe", "Aldoc", "Slc6a11", "Aqp4"],
    "Microglia": ["C1qb", "Ctss", "Sparc", "Gsn"],
    "VLMC": ["Col1a1", "Col1a2", "Lum", "Dcn"],
    "Choroid plexus": ["Ttr", "Clic6", "Igfbp2", "Krt18"],
    "Pan-neuron": ["Rbfox3", "Snap25", "Syt1", "Stmn2"],
}

marker_panel_sub = {
    celltype: [gene for gene in genes if gene in adata_sub.var_names]
    for celltype, genes in marker_panel.items()
}
sc.pl.dotplot(adata_sub, marker_panel_sub, groupby="cell_type", standard_scale="var")

sc.pl.rank_genes_groups_dotplot(adata_sub, groupby="leiden", standard_scale="var", n_genes=5)

adata_sub.write_h5ad(output_dir + "adata_sub_harmony_annotated.h5ad")
