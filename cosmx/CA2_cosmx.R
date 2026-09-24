#cosmx

library(Seurat)
library(dplyr)
library(ggplot2)
library(ggsci)
library(tidyr)
library(tibble)
library(reshape2)

fig_path = "./CosMx/CA/figure/"

theme_cell <- function(base_size = 12) {
  theme_minimal(base_size = base_size, base_family = "Arial") +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(color = "grey90", linewidth = 0.3),
      axis.line          = element_line(color = "black", linewidth = 1),
      axis.ticks         = element_line(color = "black", linewidth = 0.3),
      axis.text          = element_text(color = "black"),
      strip.text         = element_text(face = "bold", size = base_size),
      strip.background   = element_blank(),
      plot.title         = element_text(size = base_size + 1, face = "bold"),
      plot.subtitle      = element_text(size = base_size, color = "grey30"),
      legend.title       = element_text(size = base_size, face = "bold"),
      legend.key.size    = unit(0.3, "cm"),
      legend.position    = "right",
    )
}

CA2 <- readRDS("seuratObject_CA2_scaled.RDS")
CA2 <- RunPCA(CA2, assay = "SCT", verbose = FALSE)
CA2 <- FindNeighbors(CA2, reduction="pca", dims=1:30)
CA2 <- FindClusters(CA2, resolution=0.3)
CA2 <- RunUMAP(CA2, reduction = "pca", dims = 1:30)
DimPlot(CA2, reduction = "umap", label = TRUE) 
de_markers <- FindAllMarkers(object = CA2, only.pos = TRUE, min.pct = 0.25)
de_markers = read.csv("CA2_seuratObject_CA2_scaled_cluster_de_markers.csv")
top5 <- de_markers %>% group_by(cluster) %>% top_n(n = 5, wt = avg_log2FC)

#CA2 <- readRDS("./data/seuratObject_CA2_scaled_cluster_de_markers_merged_clusters_recluster.RDS")
Idents(CA2) = "seurat_cluster_res0.2"
DimPlot(CA2, reduction = "umap", label = TRUE, group.by = "seurat_cluster_res0.2") + theme_cell(base_size = 12)


# Supp Fig 6
celltype_markers <- list(
  Endothelial = c("VWF", "PECAM1", "ENG", "EMCN", "CD93", "ESAM"),
  SMC = c("ACTA2", "TAGLN", "MYH11", "CNN1", "MYOCD", "FN1", "DES","CRYAB"), # , "SYNPO2"
  Pericyte = c("RGS5", "MCAM", "CSPG4", "KCNJ8", "HIGD1B", "ABCC9", "ATP1A2"),
  Erythrocytes = c("HBB", "HBA1/2", "CD235a", "CD36"),
  Fibroblast = c("COL1A1", "COL3A1", "PDGFRA", "PDGFRB", "THY1", "FAP", "COL5A1", "LUM", "MMP2", "DCN", "APOD"),
  B = c("BLNK", "CD19", "CD79A", "CD79B", "MS4A1"),
  Plasma = c("IGHD", "IGKC", "MZB1", "JCHAIN", "SDC1", "CD38"),
  CD8T = c("CD8A", "CD8B", "CD3D"),
  CD4T = c("CD4", "CCR7", "FOXP3", "IL7R"),
  NK = c("XCL2", "B3GAT1", "FCGR3A", "GNLY", "NKG7", "NCAM1"),
  Macrophage = c("CD68", "CD14", "CD163", "CSF1R", "ITGAX"), # "FCGR3A",
  Mast = c("HPGD", "TPSB2", "HDC", "SLC18A2", "CPA3", "SLC8A3"), #"KIT", 
  DC = c("CD1C", "CD209", "CD83", "HLA-DQA1", "FCER1A", "LILRA4", "CLEC9A", "XCR1", "CADM1"),
  Neutrophil = c("CSF3R", "FCGR3B", "S100A8", "S100A9", "FPR1", "NAMPT", "MNDA"),
  Basel = c("KRT5", "KRT14", "KRT17"),
  Luminal = c("KRT8", "KRT18", "KRT19"),
  Luminal_progenitors = c("KIT", "ELF5", "ALDH1A3", "PROM1", "SLPI"),
  Mature_luminal = c("ESR1", "PGR", "GATA3", "FOXA1"),
  HER2 = c("ERBB2","GRB7","FGFR4"),
  Prolif = c("MKI67","CCNB1","CDC20","BIRC5","UBE2C")
)


celltype_markers_present <- lapply(celltype_markers, function(genes) intersect(genes, rownames(CA2)))
celltype_markers_present <- celltype_markers_present[sapply(celltype_markers_present, length) > 0]

p <- DotPlot(CA2, features = celltype_markers_present, group.by = "seurat_cluster_res0.2")
dp_data <- p$data
colnames(dp_data)   # confirm the exact column names in your Seurat version before the next step —
                     # look for "features.plot", "id", "avg.exp.scaled", "pct.exp", and the
                     # category column (usually "feature.groups" when features is a named list)

psup6b <- ggplot(dp_data, aes(x = id, y = features.plot, size = pct.exp, color = avg.exp.scaled)) +
  geom_point() +
  scale_size(range = c(0, 3), name = "Percent\nExpressed") +
  scale_color_gradient2(low = "#2a78d6", mid = "white", high = "#e34948", midpoint = 0,
                         name = "Avg. expr.\n(scaled)") +
  facet_grid(feature.groups ~ ., scales = "free_y", space = "free_y") +
  theme_minimal(base_size = 8) +
  theme(strip.text.y = element_text(angle = 0, hjust = 0),
        axis.text.x  = element_text(angle = 45, hjust = 1)) +
  labs(x = "Cluster", y = NULL)

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_marker_dotplot.png"),
  plot = psup6b,
  width = 12,
  height = 10,
  units = "in"
)

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_marker_dotplot.pdf"),
  plot = psup6b,
  width = 12,
  height = 10,
  units = "in"
)

cluster_mapping <- c(
  "0"  = "Tumor (Lum-Basal)", # yes
  "1"  = "CAFs", # yes
  "2"  = "Tumor (LumA)", # 
  "3"  = "Tumor (Hypoxic)", # 
  "4"  = "Tumor (LumB-HER2+)", # 
  "5"  = "Macrophage", # 
  "6"  = "Myeloid", # yes
  "7"  = "Endothelial", # yes
  "8" = "Erythrocytes", #
  "9" = "NK", # yes 
  "10"  = "Non-specific", #
  "11"  = "Non-specific", #  
  "12" = "Plasma" # yes
)


cols <- c(
  "CAFs"            = "#3247e8",  # Dark Blue      (Cluster 0)
  "Tumor (Lum-Basal)"   = "#FF7F0EFF",  # Light Blue     (Cluster 1)
  "Tumor (Hypoxic)" = "#fcee56",  # Light Green    (Clusters 5, 8)
  "Tumor (LumA)" = "#239b23",  # Dark Red       (Cluster 6)
  "Tumor (LumB-HER2+)" = "#254527",  # Teal-Green     (Cluster 4)
  "Non-specific"    = "#edf2ed",  # Dark Orange    (Cluster 2)
  "Macrophages"     = "#9467BDFF",  # Dark Green     (Clusters 4, 7)
  "NK"              = "#AEC7E8FF",  # Dark Brown     (Cluster 12)
  "Endothelial"     = "#6e0936",  # Dark Purple    (Clusters 10, 14, 17)
  "Erythrocytes"    = "#1d0d9b",  # Light Purple   (Cluster 11)
  "DC"              = "#98DF8AFF",  # Dark Pink      (Clusters 15, 18)
  "Plasma"          = "#C5B0D5FF",   # Dark Gray      (Cluster 20)
  "Myeloid"          = "#FF9896FF"   
)

Idents(CA2) = "seurat_cluster_res0.2"
CA2 <- Seurat::RenameIdents(CA2, cluster_mapping)
CA2@meta.data$cell_type_v2 <- Seurat::Idents(CA2)


# figure 5C
p5c <- DimPlot(
  CA2,
  reduction = "umap",
  label = FALSE,
  repel = TRUE,
  pt.size = 0.5,
  alpha = 0.5,
  cols = cols
) +
  theme_classic(base_size = 28) 

p5c

# save the png of the figure
ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_umap.png"),
  plot = p5c,
  width = 10,
  height = 6,
  units = "in"
)

# figure 5E
celltypes <- levels(factor(CA2@meta.data$cell_type_v2))

fov <- CA2@images$CA2
coords <- GetTissueCoordinates(fov)
head(coords)
df <- as.data.frame(coords)
df['cell_type'] <- CA2@meta.data$cell_type_v2


df_arranged <- df %>%
  mutate(plot_tier = case_when(
    cell_type == "Non-specific"    ~ 1,   # Rendered first (bottom background)
    cell_type == "CAFs"            ~ 4,   # Rendered after the middle layer, before Tumor
    cell_type == "Tumor (Hypoxic)" ~ 5,   # Rendered near-last (top foreground)
    cell_type == "Tumor (Lum-Basal)"   ~ 6,   # Rendered absolute last (topmost)
    grepl("^Tumor", cell_type)     ~ 3,   # Any other Tumor subtype (e.g. Luminal) — middle-late
    TRUE                           ~ 2    # Everything else (immune/stroma)
  )) %>%
  arrange(plot_tier)                      # Sort ascending: 1 -> 2 -> 3 -> 4 -> 5 -> 6

p5e <- ggplot(df_arranged, aes(x = x, y = y, color = cell_type)) +
geom_point(size = 0.01, alpha = 0.5) +
scale_color_manual(values = cols) +
coord_fixed() +
theme_classic(base_size = 14)+
    theme(
    axis.title  = element_text(size = 18),             # small axis labels
    axis.text   = element_text(size = 14, color = "black"),
    axis.line   = element_line(color = "black", linewidth = 1),  # thin axis lines
    axis.ticks  = element_line(color = "black", linewidth = 0.25),  # thin ticks
    axis.ticks.length = unit(1, "pt"),
    plot.title  = element_text(size = 8, hjust = 0.5),
  ) +
  guides(color = guide_legend(override.aes = list(size = 1.5, alpha = 1)))

p5e

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_spatial_coordinates.png"),
  plot = p5e,
  width = 9,
  height = 6,
  units = "in",
  dpi = 300
)


# Figure 5G
df$clone = CA2@meta.data$clone

clone_colors <- c(
  "clone1" = "#EBBCBAFF",
  "clone2" = "#F6C177FF",
  "clone3" = "#EB6F92FF",
  "Outside_Clones" = "#9CCFD8FF",
  "clone3_inside" = "#B5EAD7FF"
)

p5g <- ggplot(df, aes(x = x, y = y, color = clone)) +
  geom_point(size = 0.01, alpha = 0.5) +
  scale_color_manual(values = clone_colors) +
  coord_fixed() +
  theme_classic(base_size = 14) +                     # small overall text, matches reference
  theme(
    axis.title  = element_text(size = 18),             # small axis labels
    axis.text   = element_text(size = 14, color = "black"),
    axis.line   = element_line(color = "black", linewidth = 1),  # thin axis lines
    axis.ticks  = element_line(color = "black", linewidth = 0.25),  # thin ticks
    axis.ticks.length = unit(1, "pt"),
    plot.title  = element_text(size = 8, hjust = 0.5),
  ) +
  guides(color = guide_legend(override.aes = list(size = 1.5, alpha = 1)))

p5g
ggsave(
  filename = paste0(fig_path, "CA2_cosmx_clone_spatial_coordinates.png"),
  plot = p5g,
  width = 9,
  height = 6,
  units = "in",
  dpi = 300
)

# Figure 5F
df$niches = as.character(CA2@meta.data$niches)

niche_colors <- c(
  "1" = "#00BFC4",  # Blue 
  "2" = "#f6b40f",  # Orange
  "3" = "#F8766D",  # Yellow
  "4" = "#aa6adb"   # Purle
)

p5f <- ggplot(df, aes(x = x, y = y, color = niches)) +
  geom_point(size = 0.01, alpha = 0.5) +
  scale_color_manual(values = niche_colors) +
  coord_fixed() +
  theme_classic(base_size = 14) +                     # small overall text, matches reference
  theme(
    axis.title  = element_text(size = 18),             # small axis labels
    axis.text   = element_text(size = 14, color = "black"),
    axis.line   = element_line(color = "black", linewidth = 1),  # thin axis lines
    axis.ticks  = element_line(color = "black", linewidth = 0.25),  # thin ticks
    axis.ticks.length = unit(1, "pt"),
    plot.title  = element_text(size = 8, hjust = 0.5),
  ) +
  guides(color = guide_legend(override.aes = list(size = 1.5, alpha = 1)))

p5f
ggsave(
  filename = paste0(fig_path, "CA2_cosmx_niche_spatial_coordinates.png"),
  plot = p5f,
  width = 9,
  height = 6,
  units = "in",
  dpi = 300
)
# Figure 5H
composition_table <- table(CA2@meta.data$cell_type_v2, CA2@meta.data$clone)
prop_table <- prop.table(composition_table, margin = 2)
prop_df <- as.data.frame(prop_table)
colnames(prop_df) <- c("CellType", "Clone", "Proportion")

change_df <- prop_df %>%
  group_by(CellType) %>%
  summarize(range_change = max(Proportion) - min(Proportion), .groups = "drop") %>%
  arrange(desc(range_change))

prop_df <- prop_df %>%
  mutate(CellType = factor(CellType, levels = rev(change_df$CellType)))

p5h <- ggplot(prop_df, aes(x = Proportion, y = CellType)) +
  geom_line(aes(group = CellType), color = "grey75", linewidth = 0.4) +
  geom_point(aes(color = Clone), size = 4, stroke = 0) +
  scale_color_manual(values = clone_colors) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                      expand = expansion(mult = c(0.02, 0.05))) +
  labs(x = "Proportion of cells", y = NULL, color = "Clone") +
  theme_cell(base_size = 14)

p5h

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_proportions_across_clones.png"),
  plot = p5h,
  width = 8,
  height = 7,
  units = "in",
  dpi = 300
)

# Figure 5I
composition_table_niche <- table(CA2@meta.data$cell_type_v2, CA2@meta.data$niches)
prop_table_niche <- prop.table(composition_table_niche, margin = 2)
prop_df_niche <- as.data.frame(prop_table_niche)
colnames(prop_df_niche) <- c("CellType", "Niche", "Proportion")

change_df_niche <- prop_df_niche %>%
  group_by(CellType) %>%
  summarize(range_change = max(Proportion) - min(Proportion), .groups = "drop") %>%
  arrange(desc(range_change))

prop_df_niche <- prop_df_niche %>%
  mutate(CellType = factor(CellType, levels = rev(change_df$CellType)))

p5i <- ggplot(prop_df_niche, aes(x = Proportion, y = CellType)) +
  geom_line(aes(group = CellType), color = "grey75", linewidth = 0.4) +
  geom_point(aes(color = Niche), size = 4, stroke = 0) +
  #scale_color_manual(values = niche_colors) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                      expand = expansion(mult = c(0.02, 0.05))) +
  labs(x = "Proportion of cells", y = NULL, color = "Niche") +
  theme_cell(base_size = 14)

p5i

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_celltype_proportions_across_niches.png"),
  plot = p5i,
  width = 8,
  height = 7,
  units = "in",
  dpi = 300
)

# Figure 5J
# 1. Subset to tumor cells only, using existing cell_type column
tumor_types <- c("Tumor (Basal)", "Tumor (Hypoxic)", "Tumor (Luminal)", "Tumor (Luminal-ER+)")
target_clones <- c("clone1", "clone2", "clone3")

# Get tumor cell barcodes AND filter to target clones, in one step
tumor_cells <- rownames(CA2@meta.data)[
  CA2@meta.data$cell_type_v2 %in% tumor_types &
  CA2@meta.data$clone %in% target_clones
]

# 2. Pull the full sparse matrix (still sparse, no densifying) then column-subset
expr_mat_full <- GetAssayData(CA2, assay = "RNA", layer = "data")
expr_mat <- expr_mat_full[, tumor_cells, drop = FALSE]
rm(expr_mat_full); gc()   # drop the full matrix from memory immediately


# 3. Get clone label per tumor cell
clone_vec <- CA2@meta.data[tumor_cells, "clone"]

stopifnot(ncol(expr_mat) == length(clone_vec))

# 4. Compute per-clone average expression on the sparse matrix
clones <- unique(clone_vec)

avg_list <- lapply(clones, function(cl) {
  cells_in_clone <- which(clone_vec == cl)
  Matrix::rowMeans(expr_mat[, cells_in_clone, drop = FALSE])
})
names(avg_list) <- clones

avg_expr <- as.data.frame(avg_list)
avg_expr$gene <- rownames(expr_mat)
rm(expr_mat); gc()

# 5. Compute magnitude of change (log2 space) between clones
clone_cols <- clones

log_expr <- avg_expr %>%
  mutate(across(all_of(clone_cols), ~log2(.x + 1)))

# 6. Identify the "driver" clone per gene (clone with the max expression)
gene_rank <- avg_expr %>%
  mutate(across(all_of(clone_cols), ~log2(.x + 1), .names = "log2_{col}")) %>%
  rowwise() %>%
  mutate(
    max_val      = max(c_across(starts_with("log2_"))),
    min_val      = min(c_across(starts_with("log2_"))),
    log2FC_range = max_val - min_val,
    driver_clone = clone_cols[which.max(c_across(starts_with("log2_")))]
  ) %>%
  ungroup() %>%
  arrange(driver_clone, desc(log2FC_range))

# 7. Take top N divergent genes PER driver clone (so each clone gets representation)
top_n_per_clone <- 5

top_genes <- gene_rank %>%
  group_by(driver_clone) %>%
  slice_max(order_by = log2FC_range, n = top_n_per_clone) %>%
  ungroup()

# 8. Set gene factor order: grouped by driver clone, ranked by magnitude within each group
gene_order <- top_genes %>%
  arrange(driver_clone, desc(log2FC_range)) %>%
  pull(gene)

# 9. Reshape for plotting
plot_df <- top_genes %>%
  dplyr::select(gene, driver_clone, all_of(clone_cols)) %>%
  mutate(across(all_of(clone_cols), ~log2(.x + 1))) %>%
  pivot_longer(cols = all_of(clone_cols), names_to = "Clone", values_to = "log2_expr") %>%
  mutate(gene = factor(gene, levels = rev(gene_order)))

# 11. Dumbbell plot, faceted by driver clone so genes cluster by which clone they mark
p5j <- ggplot(plot_df, aes(x = log2_expr, y = gene)) +
  geom_line(aes(group = gene), color = "grey75", linewidth = 0.4) +
  geom_point(aes(color = Clone), size = 4, stroke = 0) +
  scale_color_manual(values = clone_colors) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.05))) +
  facet_grid(driver_clone ~ ., scales = "free_y", space = "free_y") +
  labs(x = expression(log[2]~"(average expression + 1)"), y = NULL, color = "Clone") +
  theme_cell()

p5j

ggsave(
  filename = paste0(fig_path, "CA2_cosmx_top_divergent_genes_across_clones.png"),
  plot = p5j,
  width = 8,
  height = 7,
  units = "in",
  dpi = 300
)