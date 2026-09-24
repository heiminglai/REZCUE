import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import norm
import os
 
# ---------------------------------------------------------------------------
# 0. Style / palette  (validated categorical + sequential + diverging ramps)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":    "#2a78d6",
    "orange":  "#eb6834",
    "aqua":    "#1baf7a",
    "red":     "#e34948",
    "gray_mid": "#f0efec",
    "ink":     "#0b0b0b",
    "ink2":    "#52514e",
    "muted":   "#FFFFFF",
    "grid":    "#e1e0d9",
    "surface": "#fcfcfb",
}
REGION_COLOR = {"R1": PALETTE["blue"], "R2": PALETTE["orange"], "R3": PALETTE["aqua"]}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
 
MM_TO_PT = 72.27 / 25.4  # ggplot2's internal mm -> pt conversion factor
CELL_BASE_SIZE = 12
CELL_GREY90 = "#e5e5e5"
CELL_GREY30 = "#4d4d4d"
CELL_FONT = [ "Arial", "DejaVu Sans"]
 
mpl.rcParams.update({
    #"figure.facecolor": PALETTE["surface"],
    #"axes.facecolor": PALETTE["surface"],
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.grid": False,
    "font.family": CELL_FONT,
    "font.size": CELL_BASE_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)
 
DATA_PATH = "protein.normalized.intensity.xls"
 
# ---------------------------------------------------------------------------
# 1. Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep="\t")
sub = df[["ID", "breast2-1", "breast2-2", "breast2-3"]].copy()
sub.columns = ["ID", "R1", "R2", "R3"]
 
detected = sub[["R1", "R2", "R3"]].notna()
sub["n_detected"] = detected.sum(axis=1)
sub = sub[sub["n_detected"] > 0].reset_index(drop=True)  # drop rows detected nowhere
 
# log2 transform (intensities are already normalized; add pseudocount for NAs->0 handling)
for c in ["R1", "R2", "R3"]:
    sub[f"log2_{c}"] = np.log2(sub[c] + 1)
 
common = sub.dropna(subset=["R1", "R2", "R3"]).copy()  # proteins detected in all 3 regions
print(f"Total proteins (detected in >=1 region): {len(sub)}")
print(f"Proteins detected in all 3 regions:       {len(common)}")


def cell_theme(ax, title=None, subtitle=None, base_size=CELL_BASE_SIZE,
                grid_axis="x", legend_ncol=1):
    """Apply the theme_cell look to a single Axes. Call this last, right
    before saving, after all data/legend calls on `ax` are done."""
    # gridlines: vertical (x) only, grey90, no minor grid -- matches
    # panel.grid.major.x set / panel.grid.major.y + panel.grid.minor blank
    ax.grid(False, which="both")
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", which="major", #color=CELL_GREY90,
                linewidth=0.3 * MM_TO_PT, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", which="major", #color=CELL_GREY90,
                linewidth=0.3 * MM_TO_PT, zorder=0)
    ax.set_axisbelow(True)
 
    # axis.line: solid black, 1mm-equivalent, bottom+left only (no boxed panel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0 * MM_TO_PT)
 
    # axis.ticks: black, thin (0.3mm); axis.text: black, unbolded, rel(0.8)*base_size
    ax.tick_params(axis="both", colors="black", width=0.3 * MM_TO_PT,
                    labelsize=round(base_size * 0.8), labelcolor="black")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("normal")
        lbl.set_color("black")
 
    # axis titles: regular weight, base_size (theme_minimal does not bold these)
    ax.xaxis.label.set_fontsize(base_size)
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontsize(base_size)
    ax.yaxis.label.set_fontweight("normal")
 
    # plot.title (bold, base_size+1) / plot.subtitle (grey30, base_size, regular)
    # -- drawn as two separate text objects so weight/color can differ, the
    # way ggplot2 treats plot.title and plot.subtitle as distinct elements.
    if title is not None:
        y0 = 1.06 if subtitle else 1.02
        ax.text(0, y0, title, transform=ax.transAxes, ha="left", va="bottom",
                 fontsize=base_size + 1, fontweight="bold", color="black")
    if subtitle is not None:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                 fontsize=base_size, fontweight="normal", color=CELL_GREY30)
 
    # legend.title (bold, base_size) / legend.key.size (small, via handle scale)
    # NOTE: legend.position="right" is NOT set here -- repositioning an
    # existing legend post-hoc requires touching matplotlib private state
    # (Legend._loc), which is fragile across versions. Create the legend
    # with loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False at
    # the call site instead; cell_theme() only restyles its text/title.
    leg = ax.get_legend()
    if leg is not None:
        if leg.get_title() is not None:
            leg.get_title().set_fontweight("bold")
            leg.get_title().set_fontsize(base_size)
        for txt in leg.get_texts():
            txt.set_fontsize(round(base_size * 0.85))
    return ax

# ---------------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------------
logv = common[["log2_R1", "log2_R2", "log2_R3"]]
logv.columns = ["R1", "R2", "R3"]
z = (logv - logv.mean()) / logv.std()
 
corr = logv.corr(method="pearson")
dist = pd.DataFrame(
    squareform(pdist(z.T.values, metric="euclidean")),
    index=["R1", "R2", "R3"], columns=["R1", "R2", "R3"]
)
 
pca = PCA(n_components=2)
pcs = pca.fit_transform(z.T.values)
pcs_df = pd.DataFrame(pcs, index=["R1", "R2", "R3"], columns=["PC1", "PC2"])
evr = pca.explained_variance_ratio_

# ---- Fig 1: PCA scatter (trajectory view) ----
fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
ax.grid(False)  # no background gridlines
for r in ["R1", "R2", "R3"]:
    ax.scatter(pcs_df.loc[r, "PC1"], pcs_df.loc[r, "PC2"], s=260,
               color=REGION_COLOR[r], edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(r, (pcs_df.loc[r, "PC1"], pcs_df.loc[r, "PC2"]),
                textcoords="offset points", xytext=(0, 16), ha="center",
                fontsize=12, fontweight="bold", color=PALETTE["ink"])
 
# bold axis lines + ticks (reference-plot style)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_linewidth(1.6)
    ax.spines[spine].set_color(PALETTE["ink"])
ax.tick_params(width=1.6, color=PALETTE["ink"], labelsize=10)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
 
fig.tight_layout()
plt.show()
fig.savefig(f"{OUTDIR}/01_pca_regions.png", dpi=300)
#plt.close(fig)



# ---------------------------------------------------------------------------
# 3. DGE MassSpec annotation retrieval
# ---------------------------------------------------------------------------
import pandas as pd
import requests
from io import StringIO

# ------------------------------------------------------------
# UniProt annotation retrieval, keyed off sub['ID'] directly
# ------------------------------------------------------------

accessions = (
    sub["ID"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print(f"Total UniProt IDs: {len(accessions)}")

batch_size = 50
annotation_results = []

for start in range(0, len(accessions), batch_size):
    batch = accessions[start:start + batch_size]
    batch_num = start // batch_size + 1
    total_batches = (len(accessions) + batch_size - 1) // batch_size
    print(f"Batch {batch_num}/{total_batches} ({len(batch)} IDs)")

    params = {
        "query": " OR ".join(f"accession:{x}" for x in batch),
        "format": "tsv",
        "fields": "accession,protein_name,gene_names",
        "size": batch_size,
    }

    try:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params=params,
            timeout=60,
        )
        if r.status_code != 200:
            print(f"  FAILED: HTTP {r.status_code}")
            print(r.text[:300])
            continue

        tmp = pd.read_csv(StringIO(r.text), sep="\t")
        annotation_results.append(tmp)
        print(f"  Retrieved {len(tmp)} annotations")

    except Exception as e:
        print(f"  ERROR: {e}")

print("Finished.")

# ------------------------------------------------------------
# Combine and merge onto sub
# ------------------------------------------------------------

annotations = pd.concat(annotation_results, ignore_index=True)

# UniProt's TSV headers are the display names, not the field IDs
annotations = annotations.rename(columns={
    "Entry": "ID",
    "Protein names": "protein_name",
    "Gene Names": "gene_names",
})

# gene_names can list several synonyms space-separated; keep primary as its own column
annotations["gene_symbol"] = (
    annotations["gene_names"].fillna("").str.split().str[0].replace("", pd.NA)
)

sub_annotated = sub.merge(
    annotations[["ID", "gene_symbol", "gene_names", "protein_name"]],
    on="ID",
    how="left",
)

sub_annotated.head()

sub_annotated = sub_annotated[[
    "ID", "R1", "R2", "R3", "n_detected",
    "log2_R1", "log2_R2", "log2_R3",
     "gene_names", "protein_name_y"
]].copy()

sub_annotated.rename(columns={
    "protein_name_y": "protein_name"
}, inplace=True)

# ---------------------------------------------------------------------------
# 4. along the R1 -> R2 -> R3 trajectory
# ---------------------------------------------------------------------------
# x-axis : endpoint-to-endpoint log2 fold change, log2(R2/R1) -- total change
#          spanned by the trajectory.
# y-axis : -log10(p) of the WEAKER of the two sequential steps
#          (R1->R2 and R2->R3), i.e. min(|z_R1_R2|, |z_R2_R3|) converted to an
#          empirical two-sided p. This is an intersection-union-style
#          statistic: a protein only scores high if BOTH legs of the
#          trajectory individually show a notable, same-direction change --
#          not just the two endpoints. Proteins where the two legs move in
#          opposite directions (non-monotonic across R1->R2->R3) are shown in
#          gray and excluded from the "gradient" call regardless of the
#          endpoint fold change.
order = ['R1', 'R2', 'R3']
o1, o2, o3 = order  # R1, R2, R3


# Recompute each leg directly from log2 intensities (sign = "increase along the
# trajectory") rather than reusing the pairwise res columns, whose a_vs_b sign
# convention doesn't line up uniformly with the o1->o2->o3 walking direction.
sub_idx = sub.set_index("ID")
diff_leg1 = sub_idx[f"log2_{o2}"] - sub_idx[f"log2_{o1}"]   # o1 -> o2 step (R1 -> R2)
diff_leg2 = sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o2}"]   # o2 -> o3 step (R2 -> R3)
z_leg1_full = (diff_leg1 - diff_leg1.mean()) / diff_leg1.std()
z_leg2_full = (diff_leg2 - diff_leg2.mean()) / diff_leg2.std()


traj = sub_annotated[["ID", "gene_names", "protein_name"]].copy()

#traj = res[["ID"]].copy()
traj["log2FC_total"] = (sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o1}"]).reindex(traj["ID"]).values
traj["z_leg1"] = z_leg1_full.reindex(traj["ID"]).values  # o1 -> o2 (R1 -> R2)
traj["z_leg2"] = z_leg2_full.reindex(traj["ID"]).values  # o2 -> o3 (R2 -> R3)
traj["same_direction"] = np.sign(traj["z_leg1"]) == np.sign(traj["z_leg2"])
traj["z_min"] = np.where(traj["same_direction"],
                          np.minimum(traj["z_leg1"].abs(), traj["z_leg2"].abs()),
                          0.0)
traj["p_trajectory"] = 2 * (1 - norm.cdf(traj["z_min"]))
traj["gradient_class"] = np.select(
    [ traj["same_direction"] & (traj["log2FC_total"] >= 1) & (traj["z_min"] >= 2),
      traj["same_direction"] & (traj["log2FC_total"] <= -1) & (traj["z_min"] >= 2) ],
    [f"gradient_up_{o1}_to_{o3}", f"gradient_down_{o1}_to_{o3}"],
    default="not_monotonic_or_ns"
)
#traj = traj.merge(sub[["ID"]], on="ID")
#traj.to_csv("gradient_trajectory_results.csv", index=False)
 
n_up = (traj["gradient_class"] == f"gradient_up_{o1}_to_{o2}").sum()
n_down = (traj["gradient_class"] == f"gradient_down_{o2}_to_{o3}").sum()

CAP = 20
negp_raw = -np.log10(traj["p_trajectory"].clip(lower=1e-300))
traj["negp_capped"] = negp_raw.clip(upper=CAP)
traj["is_capped"] = negp_raw > CAP
 
def repel_labels(ax, xy_data, texts, min_dx_px=95, min_dy_px=26, iterations=400, offset_px=12):
    """Lightweight, dependency-free label-repulsion: treats each label as a
    small rectangle (min_dx_px x min_dy_px, sized for ~6-character IDs at the
    plotted font size) and resolves overlaps via axis-aligned box separation
    -- pushes apart along whichever axis (x or y) has the smaller overlap, so
    labels spread mostly sideways rather than needlessly far vertically.
    Returns final data-space label coordinates for ax.annotate. Small point
    counts only (<~50); O(n^2) per iteration."""
    pts_px = ax.transData.transform(np.asarray(xy_data, dtype=float))
    lab_px = pts_px.copy()
    lab_px[:, 1] += offset_px  # start just above each point
    rng = np.random.default_rng(0)
    lab_px += rng.uniform(-0.5, 0.5, size=lab_px.shape)  # break exact ties
    n = len(pts_px)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = lab_px[i, 0] - lab_px[j, 0]
                dy = lab_px[i, 1] - lab_px[j, 1]
                if abs(dx) < min_dx_px and abs(dy) < min_dy_px:
                    moved = True
                    overlap_x = min_dx_px - abs(dx)
                    overlap_y = min_dy_px - abs(dy)
                    if overlap_x < overlap_y:
                        shift = overlap_x / 2 + 0.5
                        sx = shift if dx >= 0 else -shift
                        lab_px[i, 0] += sx
                        lab_px[j, 0] -= sx
                    else:
                        shift = overlap_y / 2 + 0.5
                        sy = shift if dy >= 0 else -shift
                        lab_px[i, 1] += sy
                        lab_px[j, 1] -= sy
            # nudge away from any nearby data point too (not just other labels)
            for j in range(n):
                if j == i:
                    continue
                dx = lab_px[i, 0] - pts_px[j, 0]
                dy = lab_px[i, 1] - pts_px[j, 1]
                bx, by = min_dx_px * 0.55, min_dy_px * 1.3
                if abs(dx) < bx and abs(dy) < by:
                    moved = True
                    overlap_x, overlap_y = bx - abs(dx), by - abs(dy)
                    if overlap_x < overlap_y:
                        lab_px[i, 0] += (overlap_x * 0.3) * (1 if dx >= 0 else -1)
                    else:
                        lab_px[i, 1] += (overlap_y * 0.3) * (1 if dy >= 0 else -1)
        if not moved:
            break
    return ax.transData.inverted().transform(lab_px)
 
 
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
colors = {f"gradient_up_{o1}_to_{o3}": PALETTE["red"],
          f"gradient_down_{o1}_to_{o3}": PALETTE["blue"],
          "not_monotonic_or_ns": "#c9c8c2"}
for lab in ["not_monotonic_or_ns", f"gradient_down_{o1}_to_{o3}", f"gradient_up_{o1}_to_{o3}"]:
    m = traj["gradient_class"] == lab
    ax.scatter(traj.loc[m, "log2FC_total"], traj.loc[m, "negp_capped"],
               s=10 if lab == "not_monotonic_or_ns" else 18,
               color=colors[lab], alpha=0.5 if lab == "not_monotonic_or_ns" else 0.85,
               linewidth=0, label=lab.replace("_", " "), zorder=2)
cap_m = traj["is_capped"]
if cap_m.any():
    ax.scatter(traj.loc[cap_m, "log2FC_total"], traj.loc[cap_m, "negp_capped"],
               s=26, facecolors="none", edgecolors=PALETTE["ink"], linewidth=0.8,
               zorder=4, label=f"capped ({cap_m.sum()} pts, see CSV)")
ax.axvline(1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.axvline(-1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.set_xlabel(f"log2 fold change", fontsize=20, fontweight="bold")
ax.set_ylabel(f"-log10(p)", fontsize=20, fontweight="bold")
ax.set_title(f"Gradient proteins along trajectory {o1} → {o2} → {o3}")
#ax.legend(frameon=False, loc="upper left", fontsize=12)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title=None)
 
# bold axis lines + ticks (same style as the PCA plot)
ax.grid(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_linewidth(1.6)
    ax.spines[spine].set_color(PALETTE["ink"])
ax.tick_params(width=1.6, color=PALETTE["ink"], labelsize=10)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
 
# label the called gradient up/down proteins (protein ID = UniProt accession;
# no gene-symbol mapping available offline) with repelled, non-overlapping labels
label_mask = traj["gradient_class"] != "not_monotonic_or_ns"
xy = traj.loc[label_mask, ["log2FC_total", "negp_capped"]].values
label_data = traj.loc[label_mask].copy()
label_ids = (
    label_data["gene_names"]
    .fillna(label_data["ID"])
    .astype(str)
    .values
)
label_pos = repel_labels(ax, xy, label_ids)
 
for (x0, y0), (lx, ly), pid, cls in zip(xy, label_pos, label_ids,
                                         traj.loc[label_mask, "gradient_class"]):
    col = colors[cls]
    ax.annotate(pid, xy=(x0, y0), xytext=(lx, ly), fontsize=14, color=PALETTE["ink"],
                ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.8,
                                 shrinkA=2, shrinkB=2))
 
fig.tight_layout()
#plt.show()
#fig.savefig(f"{OUTDIR}/09_gradient_volcano_trajectory.png", dpi=300)
fig.savefig(f"{OUTDIR}/09_CA2_R123_gradient_volcano_trajectory.png", dpi=300)
plt.close(fig)


# ---------------------------------------------------------------------------
# 5. along the R3 -> R2 -> R1 trajectory
# ---------------------------------------------------------------------------
# x-axis : endpoint-to-endpoint log2 fold change, log2(R2/R1) -- total change
#          spanned by the trajectory.
# y-axis : -log10(p) of the WEAKER of the two sequential steps
#          (R3->R2 and R2->R1), i.e. min(|z_R3_R2|, |z_R2_R1|) converted to an
#          empirical two-sided p. This is an intersection-union-style
#          statistic: a protein only scores high if BOTH legs of the
#          trajectory individually show a notable, same-direction change --
#          not just the two endpoints. Proteins where the two legs move in
#          opposite directions (non-monotonic across R1->R2->R3) are shown in
#          gray and excluded from the "gradient" call regardless of the
#          endpoint fold change.
order = [ 'R3','R2','R1']
o1, o2, o3 = order  # R3, R2, R1


# Recompute each leg directly from log2 intensities (sign = "increase along the
# trajectory") rather than reusing the pairwise res columns, whose a_vs_b sign
# convention doesn't line up uniformly with the o1->o2->o3 walking direction.
sub_idx = sub.set_index("ID")
diff_leg1 = sub_idx[f"log2_{o2}"] - sub_idx[f"log2_{o1}"]   # o1 -> o2 step (R1 -> R2)
diff_leg2 = sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o2}"]   # o2 -> o3 step (R2 -> R3)
z_leg1_full = (diff_leg1 - diff_leg1.mean()) / diff_leg1.std()
z_leg2_full = (diff_leg2 - diff_leg2.mean()) / diff_leg2.std()


traj = sub_annotated[["ID", "gene_names", "protein_name"]].copy()

#traj = res[["ID"]].copy()
traj["log2FC_total"] = (sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o1}"]).reindex(traj["ID"]).values
traj["z_leg1"] = z_leg1_full.reindex(traj["ID"]).values  # o1 -> o2 (R1 -> R2)
traj["z_leg2"] = z_leg2_full.reindex(traj["ID"]).values  # o2 -> o3 (R2 -> R3)
traj["same_direction"] = np.sign(traj["z_leg1"]) == np.sign(traj["z_leg2"])
traj["z_min"] = np.where(traj["same_direction"],
                          np.minimum(traj["z_leg1"].abs(), traj["z_leg2"].abs()),
                          0.0)
traj["p_trajectory"] = 2 * (1 - norm.cdf(traj["z_min"]))
traj["gradient_class"] = np.select(
    [ traj["same_direction"] & (traj["log2FC_total"] >= 1) & (traj["z_min"] >= 2),
      traj["same_direction"] & (traj["log2FC_total"] <= -1) & (traj["z_min"] >= 2) ],
    [f"gradient_up_{o1}_to_{o3}", f"gradient_down_{o1}_to_{o3}"],
    default="not_monotonic_or_ns"
)
#traj = traj.merge(sub[["ID"]], on="ID")
#traj.to_csv("gradient_trajectory_results.csv", index=False)
 
n_up = (traj["gradient_class"] == f"gradient_up_{o1}_to_{o2}").sum()
n_down = (traj["gradient_class"] == f"gradient_down_{o2}_to_{o3}").sum()

CAP = 20
negp_raw = -np.log10(traj["p_trajectory"].clip(lower=1e-300))
traj["negp_capped"] = negp_raw.clip(upper=CAP)
traj["is_capped"] = negp_raw > CAP
 
def repel_labels(ax, xy_data, texts, min_dx_px=95, min_dy_px=26, iterations=400, offset_px=12):
    """Lightweight, dependency-free label-repulsion: treats each label as a
    small rectangle (min_dx_px x min_dy_px, sized for ~6-character IDs at the
    plotted font size) and resolves overlaps via axis-aligned box separation
    -- pushes apart along whichever axis (x or y) has the smaller overlap, so
    labels spread mostly sideways rather than needlessly far vertically.
    Returns final data-space label coordinates for ax.annotate. Small point
    counts only (<~50); O(n^2) per iteration."""
    pts_px = ax.transData.transform(np.asarray(xy_data, dtype=float))
    lab_px = pts_px.copy()
    lab_px[:, 1] += offset_px  # start just above each point
    rng = np.random.default_rng(0)
    lab_px += rng.uniform(-0.5, 0.5, size=lab_px.shape)  # break exact ties
    n = len(pts_px)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = lab_px[i, 0] - lab_px[j, 0]
                dy = lab_px[i, 1] - lab_px[j, 1]
                if abs(dx) < min_dx_px and abs(dy) < min_dy_px:
                    moved = True
                    overlap_x = min_dx_px - abs(dx)
                    overlap_y = min_dy_px - abs(dy)
                    if overlap_x < overlap_y:
                        shift = overlap_x / 2 + 0.5
                        sx = shift if dx >= 0 else -shift
                        lab_px[i, 0] += sx
                        lab_px[j, 0] -= sx
                    else:
                        shift = overlap_y / 2 + 0.5
                        sy = shift if dy >= 0 else -shift
                        lab_px[i, 1] += sy
                        lab_px[j, 1] -= sy
            # nudge away from any nearby data point too (not just other labels)
            for j in range(n):
                if j == i:
                    continue
                dx = lab_px[i, 0] - pts_px[j, 0]
                dy = lab_px[i, 1] - pts_px[j, 1]
                bx, by = min_dx_px * 0.55, min_dy_px * 1.3
                if abs(dx) < bx and abs(dy) < by:
                    moved = True
                    overlap_x, overlap_y = bx - abs(dx), by - abs(dy)
                    if overlap_x < overlap_y:
                        lab_px[i, 0] += (overlap_x * 0.3) * (1 if dx >= 0 else -1)
                    else:
                        lab_px[i, 1] += (overlap_y * 0.3) * (1 if dy >= 0 else -1)
        if not moved:
            break
    return ax.transData.inverted().transform(lab_px)
 
 
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
colors = {f"gradient_up_{o1}_to_{o3}": PALETTE["red"],
          f"gradient_down_{o1}_to_{o3}": PALETTE["blue"],
          "not_monotonic_or_ns": "#c9c8c2"}
for lab in ["not_monotonic_or_ns", f"gradient_down_{o1}_to_{o3}", f"gradient_up_{o1}_to_{o3}"]:
    m = traj["gradient_class"] == lab
    ax.scatter(traj.loc[m, "log2FC_total"], traj.loc[m, "negp_capped"],
               s=10 if lab == "not_monotonic_or_ns" else 18,
               color=colors[lab], alpha=0.5 if lab == "not_monotonic_or_ns" else 0.85,
               linewidth=0, label=lab.replace("_", " "), zorder=2)
cap_m = traj["is_capped"]
if cap_m.any():
    ax.scatter(traj.loc[cap_m, "log2FC_total"], traj.loc[cap_m, "negp_capped"],
               s=26, facecolors="none", edgecolors=PALETTE["ink"], linewidth=0.8,
               zorder=4, label=f"capped ({cap_m.sum()} pts, see CSV)")
ax.axvline(1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.axvline(-1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.set_xlabel(f"log2 fold change", fontsize=20, fontweight="bold")
ax.set_ylabel(f"-log10(p)", fontsize=20, fontweight="bold")
ax.set_title(f"Gradient proteins along trajectory {o1} → {o2} → {o3}")
#ax.legend(frameon=False, loc="upper left", fontsize=12)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title=None)
 
# bold axis lines + ticks (same style as the PCA plot)
ax.grid(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_linewidth(1.6)
    ax.spines[spine].set_color(PALETTE["ink"])
ax.tick_params(width=1.6, color=PALETTE["ink"], labelsize=10)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
 
# label the called gradient up/down proteins (protein ID = UniProt accession;
# no gene-symbol mapping available offline) with repelled, non-overlapping labels
label_mask = traj["gradient_class"] != "not_monotonic_or_ns"
xy = traj.loc[label_mask, ["log2FC_total", "negp_capped"]].values
label_data = traj.loc[label_mask].copy()
label_ids = (
    label_data["gene_names"]
    .fillna(label_data["ID"])
    .astype(str)
    .values
)
label_pos = repel_labels(ax, xy, label_ids)
 
for (x0, y0), (lx, ly), pid, cls in zip(xy, label_pos, label_ids,
                                         traj.loc[label_mask, "gradient_class"]):
    col = colors[cls]
    ax.annotate(pid, xy=(x0, y0), xytext=(lx, ly), fontsize=14, color=PALETTE["ink"],
                ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.8,
                                 shrinkA=2, shrinkB=2))
 
fig.tight_layout()
plt.show()
fig.savefig(f"{OUTDIR}/09_CA2_R321_gradient_volcano_trajectory.png", dpi=300)
#plt.close(fig)


# ---------------------------------------------------------------------------
# 6. along the R1 -> R3 -> R2 trajectory
# ---------------------------------------------------------------------------
# x-axis : endpoint-to-endpoint log2 fold change, log2(R2/R1) -- total change
#          spanned by the trajectory.
# y-axis : -log10(p) of the WEAKER of the two sequential steps
#          (R1->R3 and R3->R2), i.e. min(|z_R1_R3|, |z_R3_R2|) converted to an
#          empirical two-sided p. This is an intersection-union-style
#          statistic: a protein only scores high if BOTH legs of the
#          trajectory individually show a notable, same-direction change --
#          not just the two endpoints. Proteins where the two legs move in
#          opposite directions (non-monotonic across R1->R2->R3) are shown in
#          gray and excluded from the "gradient" call regardless of the
#          endpoint fold change.
order = ['R1', 'R3', 'R2']
o1, o2, o3 = order  # R1, R3, R2


# Recompute each leg directly from log2 intensities (sign = "increase along the
# trajectory") rather than reusing the pairwise res columns, whose a_vs_b sign
# convention doesn't line up uniformly with the o1->o2->o3 walking direction.
sub_idx = sub.set_index("ID")
diff_leg1 = sub_idx[f"log2_{o2}"] - sub_idx[f"log2_{o1}"]   # o1 -> o2 step (R1 -> R3)
diff_leg2 = sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o2}"]   # o2 -> o3 step (R3 -> R2)
z_leg1_full = (diff_leg1 - diff_leg1.mean()) / diff_leg1.std()
z_leg2_full = (diff_leg2 - diff_leg2.mean()) / diff_leg2.std()


traj = sub_annotated[["ID", "gene_names", "protein_name"]].copy()

#traj = res[["ID"]].copy()
traj["log2FC_total"] = (sub_idx[f"log2_{o3}"] - sub_idx[f"log2_{o1}"]).reindex(traj["ID"]).values
traj["z_leg1"] = z_leg1_full.reindex(traj["ID"]).values  # o1 -> o2 (R1 -> R3)
traj["z_leg2"] = z_leg2_full.reindex(traj["ID"]).values  # o2 -> o3 (R3 -> R2)
traj["same_direction"] = np.sign(traj["z_leg1"]) == np.sign(traj["z_leg2"])
traj["z_min"] = np.where(traj["same_direction"],
                          np.minimum(traj["z_leg1"].abs(), traj["z_leg2"].abs()),
                          0.0)
traj["p_trajectory"] = 2 * (1 - norm.cdf(traj["z_min"]))
traj["gradient_class"] = np.select(
    [ traj["same_direction"] & (traj["log2FC_total"] >= 1) & (traj["z_min"] >= 2),
      traj["same_direction"] & (traj["log2FC_total"] <= -1) & (traj["z_min"] >= 2) ],
    [f"gradient_up_{o1}_to_{o3}", f"gradient_down_{o1}_to_{o3}"],
    default="not_monotonic_or_ns"
)
#traj = traj.merge(sub[["ID"]], on="ID")
#traj.to_csv("gradient_trajectory_results.csv", index=False)
 
n_up = (traj["gradient_class"] == f"gradient_up_{o1}_to_{o3}").sum()
n_down = (traj["gradient_class"] == f"gradient_down_{o1}_to_{o3}").sum()

CAP = 20
negp_raw = -np.log10(traj["p_trajectory"].clip(lower=1e-300))
traj["negp_capped"] = negp_raw.clip(upper=CAP)
traj["is_capped"] = negp_raw > CAP
 
def repel_labels(ax, xy_data, texts, min_dx_px=95, min_dy_px=26, iterations=400, offset_px=12):
    """Lightweight, dependency-free label-repulsion: treats each label as a
    small rectangle (min_dx_px x min_dy_px, sized for ~6-character IDs at the
    plotted font size) and resolves overlaps via axis-aligned box separation
    -- pushes apart along whichever axis (x or y) has the smaller overlap, so
    labels spread mostly sideways rather than needlessly far vertically.
    Returns final data-space label coordinates for ax.annotate. Small point
    counts only (<~50); O(n^2) per iteration."""
    pts_px = ax.transData.transform(np.asarray(xy_data, dtype=float))
    lab_px = pts_px.copy()
    lab_px[:, 1] += offset_px  # start just above each point
    rng = np.random.default_rng(0)
    lab_px += rng.uniform(-0.5, 0.5, size=lab_px.shape)  # break exact ties
    n = len(pts_px)
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = lab_px[i, 0] - lab_px[j, 0]
                dy = lab_px[i, 1] - lab_px[j, 1]
                if abs(dx) < min_dx_px and abs(dy) < min_dy_px:
                    moved = True
                    overlap_x = min_dx_px - abs(dx)
                    overlap_y = min_dy_px - abs(dy)
                    if overlap_x < overlap_y:
                        shift = overlap_x / 2 + 0.5
                        sx = shift if dx >= 0 else -shift
                        lab_px[i, 0] += sx
                        lab_px[j, 0] -= sx
                    else:
                        shift = overlap_y / 2 + 0.5
                        sy = shift if dy >= 0 else -shift
                        lab_px[i, 1] += sy
                        lab_px[j, 1] -= sy
            # nudge away from any nearby data point too (not just other labels)
            for j in range(n):
                if j == i:
                    continue
                dx = lab_px[i, 0] - pts_px[j, 0]
                dy = lab_px[i, 1] - pts_px[j, 1]
                bx, by = min_dx_px * 0.55, min_dy_px * 1.3
                if abs(dx) < bx and abs(dy) < by:
                    moved = True
                    overlap_x, overlap_y = bx - abs(dx), by - abs(dy)
                    if overlap_x < overlap_y:
                        lab_px[i, 0] += (overlap_x * 0.3) * (1 if dx >= 0 else -1)
                    else:
                        lab_px[i, 1] += (overlap_y * 0.3) * (1 if dy >= 0 else -1)
        if not moved:
            break
    return ax.transData.inverted().transform(lab_px)
 
 
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
colors = {f"gradient_up_{o1}_to_{o3}": PALETTE["red"],
          f"gradient_down_{o1}_to_{o3}": PALETTE["blue"],
          "not_monotonic_or_ns": "#c9c8c2"}
for lab in ["not_monotonic_or_ns", f"gradient_down_{o1}_to_{o3}", f"gradient_up_{o1}_to_{o3}"]:
    m = traj["gradient_class"] == lab
    ax.scatter(traj.loc[m, "log2FC_total"], traj.loc[m, "negp_capped"],
               s=10 if lab == "not_monotonic_or_ns" else 18,
               color=colors[lab], alpha=0.5 if lab == "not_monotonic_or_ns" else 0.85,
               linewidth=0, label=lab.replace("_", " "), zorder=2)
cap_m = traj["is_capped"]
if cap_m.any():
    ax.scatter(traj.loc[cap_m, "log2FC_total"], traj.loc[cap_m, "negp_capped"],
               s=26, facecolors="none", edgecolors=PALETTE["ink"], linewidth=0.8,
               zorder=4, label=f"capped ({cap_m.sum()} pts, see CSV)")
ax.axvline(1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.axvline(-1, color=PALETTE["muted"], linestyle=":", linewidth=1)
ax.set_xlabel(f"log2 fold change", fontsize=20, fontweight="bold")
ax.set_ylabel(f"-log10(p)", fontsize=20, fontweight="bold")
ax.set_title(f"Gradient proteins along trajectory {o1} → {o2} → {o3}")
#ax.legend(frameon=False, loc="upper left", fontsize=12)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title=None)
 
# bold axis lines + ticks (same style as the PCA plot)
ax.grid(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_linewidth(1.6)
    ax.spines[spine].set_color(PALETTE["ink"])
ax.tick_params(width=1.6, color=PALETTE["ink"], labelsize=10)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
 
# label the called gradient up/down proteins (protein ID = UniProt accession;
# no gene-symbol mapping available offline) with repelled, non-overlapping labels
label_mask = traj["gradient_class"] != "not_monotonic_or_ns"
xy = traj.loc[label_mask, ["log2FC_total", "negp_capped"]].values
label_data = traj.loc[label_mask].copy()
label_ids = (
    label_data["gene_names"]
    .fillna(label_data["ID"])
    .astype(str)
    .values
)
label_pos = repel_labels(ax, xy, label_ids)
 
for (x0, y0), (lx, ly), pid, cls in zip(xy, label_pos, label_ids,
                                         traj.loc[label_mask, "gradient_class"]):
    col = colors[cls]
    ax.annotate(pid, xy=(x0, y0), xytext=(lx, ly), fontsize=10, color=PALETTE["ink"],
                ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.8,
                                 shrinkA=2, shrinkB=2))
 
fig.tight_layout()
plt.show()
#
fig.savefig(f"{OUTDIR}/09_CA2_R132_gradient_volcano_trajectory.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Venn diagram of protein/peptide ID overlap across R1, R2, R3
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

R1_ids = set(sub_annotated.query("R1.notna()")["gene_names"])
R2_ids = set(sub_annotated.query("R2.notna()")["gene_names"])
R3_ids = set(sub_annotated.query("R3.notna()")["gene_names"])

fig, ax = plt.subplots(figsize=(5,5))

v = venn3(
    [R1_ids, R2_ids, R3_ids],
    set_labels=("R1", "R2", "R3"),
    ax=ax
)

# Adjust font sizes
for text in v.set_labels:
    if text:
        text.set_fontsize(14)

for text in v.subset_labels:
    if text:
        text.set_fontsize(12)

plt.title(
    "Protein/Peptide ID Overlap",
    fontsize=16,
    pad=20
)

plt.tight_layout()
#plt.show()

fig.savefig(f"{OUTDIR}/fig5k_CA2_venn.png", dpi=300)
plt.close(fig)