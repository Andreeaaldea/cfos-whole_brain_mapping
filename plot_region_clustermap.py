
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# ----------------- CONFIG -----------------
effects_xlsx = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\effect_sizes_PE_vs_CT_by_hemi_with_categories.xlsx"
out_png_dir  = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\ES_heatmaps"
top_n        = 160                 # number of regions (columns) to show
metric       = "g"                # or "d" (we'll cluster/plot this)
rows_to_use  = ["L", "R", "Delta"]  # rows of the heatmap
ref_genotype = "WT"               # use WT order as reference (so WT & Shank3 are comparable)
other_geno   = "Shank3"           # also plot using the same column order
USE_ROW_ZSCORE_FOR_DISPLAY = False   # keep False for honest “novel vs familiar”
METRIC_NAME = "g"                    # or "d"
DELTA_FORMULA = "R-L"  # choose "R-L" (default) or "L-R"
DELTA_LABEL   = f"Delta ({DELTA_FORMULA})"

# ------------------------------------------

Path(out_png_dir).mkdir(parents=True, exist_ok=True)

def load_sheet_like(xlsx, target_prefix):
    xl = pd.ExcelFile(xlsx)
    pat = re.compile(rf"^{re.escape(target_prefix)}($|[_\-])", re.I)
    for name in xl.sheet_names:
        if pat.match(name):
            return pd.read_excel(xlsx, sheet_name=name)
    return None

def ensure_category(df):
    if "category" in df.columns:
        return df
    if "structure_id_path" in df.columns:
        category_map = {
            "Medulla": "/997/8/343/1065/354/",
            "Pons": "/997/8/343/1065/771/",
            "Hypothalamus": "/997/8/343/1129/1097/",
            "Thalamus": "/997/8/343/1129/549/",
            "Midbrain": "/997/8/343/313/",
            "Cerebellum": "/997/8/512/",
            "Cortical plate": "/997/8/567/688/695/",
            "Cortical subplate": "/997/8/567/688/703/",
            "Pallidum": "/997/8/567/623/803/",
            "Striatum": "/997/8/567/623/477/",
        }
        def assign_cat(p):
            p = "" if pd.isna(p) else str(p)
            for cat, pref in category_map.items():
                if p.startswith(pref):
                    return cat
            return "Other"
        out = df.copy()
        out["category"] = out["structure_id_path"].map(assign_cat)
        return out
    out = df.copy()
    out["category"] = "Other"
    return out

def compute_delta_sheet(dfL, dfR, metric="g", sign="R-L"):
    """
    Build a Delta dataframe from L & R with columns:
      ['region_id','acronym','name','category', metric]
    where `metric` holds Delta = (R-L) or (L-R).
    Returns None if either L or R is missing.
    """
    if dfL is None or dfR is None:
        return None

    # Decide join key
    key = "region_id" if ("region_id" in dfL.columns and "region_id" in dfR.columns) else None

    # Minimal columns to carry through
    colsL = [c for c in ["region_id","acronym","name","category",metric] if c in dfL.columns]
    colsR = [c for c in ["region_id","acronym","name","category",metric] if c in dfR.columns]
    L = dfL[colsL].rename(columns={metric: "L"})
    R = dfR[colsR].rename(columns={metric: "R"})

    if key:
        # avoid duplicating meta columns from R on merge
        dropR = [c for c in ["acronym","name","category"] if c in R.columns]
        M = pd.merge(L, R.drop(columns=dropR), on=key, how="outer")
    else:
        M = pd.merge(L, R, on=["acronym","name","category"], how="outer")

    # Compute Delta
    if sign.upper() == "R-L":
        M["Delta"] = M["R"] - M["L"]
    else:
        M["Delta"] = M["L"] - M["R"]

    out = M.copy()
    out[metric] = out["Delta"]
    keep = [c for c in ["region_id","acronym","name","category",metric] if c in out.columns]
    return out[keep]

def build_matrix(df_L, df_R, df_D, metric="g"):
    """
    Return:
      M  : rows x cols matrix (rows in order L, R, Delta where available)
      rows: row labels used
      cols: list of (region_id, label) tuples in the chosen order
      categories: list of category strings (aligned to cols)
      labels: human-friendly column labels (acronym > name > id)
    """
    # Decide the join key from whichever df has it
    key = None
    for d in (df_L, df_R, df_D):
        if d is not None and "region_id" in d.columns:
            key = "region_id"
            break

    def pick_label_row(r):
        for k in ("acronym","name","region_id"):
            if k in r and pd.notna(r[k]):
                return str(r[k])
        return str(r.name)

    # Build a wide frame with columns for each row we want
    dfs = {}
    if df_L is not None: dfs["L"] = df_L[[key,"acronym","name","category",metric]] if key else df_L[["acronym","name","category",metric]]
    if df_R is not None: dfs["R"] = df_R[[key,"acronym","name","category",metric]] if key else df_R[["acronym","name","category",metric]]
    if df_D is not None: dfs["Delta"] = df_D[[key,"acronym","name","category",metric]] if key else df_D[["acronym","name","category",metric]]

    # Use L as base for labels/categories; otherwise whichever exists
    base_key = next(iter(dfs))
    base = dfs[base_key].copy()
    base = base.rename(columns={metric: f"{metric}_{base_key}"})

    for rname, d in dfs.items():
        if rname == base_key: continue
        d = d.rename(columns={metric: f"{metric}_{rname}"})
        if key:
            base = base.merge(d.drop(columns=[c for c in ["acronym","name","category"] if c in d.columns]),
                              on=key, how="outer")
        else:
            base = base.merge(d, on=["acronym","name","category"], how="outer")

    # pick a display label
    base["__label__"] = base.apply(pick_label_row, axis=1)

    # choose top-N columns by the max absolute effect across rows we will plot
    row_cols = [f"{metric}_{r}" for r in ["L","R","Delta"] if f"{metric}_{r}" in base.columns]
    base["__rank_val__"] = np.nanmax(np.abs(base[row_cols].to_numpy(float)), axis=1)
    base = base.sort_values("__rank_val__", ascending=False).head(top_n)

    # build M: rows x cols
    cols_order = list(base["__label__"])
    rows_present = [r for r in ["L","R","Delta"] if f"{metric}_{r}" in base.columns]
    M = []
    for r in rows_present:
        M.append(base[f"{metric}_{r}"].to_numpy(float))
    M = np.vstack(M) if M else np.zeros((0, len(cols_order)))

    # categories aligned to columns (from base)
    categories = base["category"].astype(str).tolist()
    # keep region_id too if present
    if key:
        region_ids = base[key].tolist()
        cols_meta = list(zip(region_ids, cols_order))
    else:
        cols_meta = list(zip([None]*len(cols_order), cols_order))

    return M, rows_present, cols_meta, categories, cols_order

def zscore_rows(M):
    M = M.copy()
    for i in range(M.shape[0]):
        row = M[i, :]
        m = np.nanmean(row); s = np.nanstd(row)
        M[i, :] = (row - m) / s if (np.isfinite(s) and s > 0) else (row - m)
    return M


def cluster_columns(M):
    # replace NaNs with row means for distance calc
    X = M.copy()
    for i in range(X.shape[0]):
        row = X[i, :]
        if np.any(~np.isfinite(row)):
            m = np.nanmean(row)
            row[~np.isfinite(row)] = m
            X[i,:] = row
    # Euclidean on row-wise z-scores, Ward gives nice compact clusters
    Z = linkage(X.T, method="ward", metric="euclidean")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    return Z, leaves

def align_rows(M, have_rows, want_rows):
    """Reorder/add rows so M has rows in want_rows order (missing rows -> NaN)."""
    have_rows = list(have_rows)
    idx = {r: i for i, r in enumerate(have_rows)}
    out = np.full((len(want_rows), M.shape[1]), np.nan, dtype=float)
    for k, r in enumerate(want_rows):
        if r in idx:
            out[k, :] = M[idx[r], :]
    return out

def align_columns(M, have_labels, want_labels):
    """Reorder/add columns so M has columns in want_labels order (missing cols -> NaN)."""
    have_labels = list(have_labels)
    idx = {c: i for i, c in enumerate(have_labels)}
    out = np.full((M.shape[0], len(want_labels)), np.nan, dtype=float)
    for j, c in enumerate(want_labels):
        if c in idx:
            out[:, j] = M[:, idx[c]]
    return out


def color_strip_from_categories(values):
    """
    Return:
      strip: 1 x N x 3 float array for imshow (RGB in [0,1])
      lut:   {category -> hex color} for the legend
    """
    vals = pd.Series(values, dtype="object").fillna("Other")
    # Stable “first appearance” unique order
    cats = vals.drop_duplicates().tolist()

    # palette (extend/repeat if needed)
    palette = [
        "#6e40aa","#4277b3","#00a1c1","#1bb182","#6bbb57",
        "#b9c33d","#ffd33d","#fca636","#f17342","#e84e63","#cf3a86"
    ]
    # map category -> color
    lut = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

    # build an RGB array (1 x N x 3) for imshow
    rgb = np.array([to_rgb(lut[c]) for c in vals.tolist()], dtype=float)[None, :, :]
    return rgb, lut


#def plot_clustermap(M, rows, cols_labels, categories, col_order, title, outpath, Z, leaves, metric):
    # order columns by dendrogram leaves
    M_ord = M[:, leaves]
    labels_ord = [col_order[i] for i in leaves]
    cats_ord   = [categories[i] for i in leaves]

    # build figure: dendrogram (top), color strip (just below), heatmap (bottom)
    h_dend  = 1.2
    h_strip = 0.25
    h_heat  = max(2.0, 0.25 * len(rows))
    total_h = h_dend + h_strip + h_heat
    fig = plt.figure(figsize=(max(10, 0.3*len(labels_ord)), total_h), constrained_layout=False)

    # axes
    ax_dend  = fig.add_axes([0.06, (h_strip+h_heat)/total_h, 0.91, h_dend/total_h])
    ax_strip = fig.add_axes([0.06, (h_heat)/total_h,          0.91, h_strip/total_h])
    ax_heat  = fig.add_axes([0.06, 0.05,                       0.91, (h_heat-0.05)/total_h])

    # dendrogram
    dendrogram(Z, ax=ax_dend, no_labels=True, color_threshold=None)
    ax_dend.set_xticks([]); ax_dend.set_yticks([])
    ax_dend.set_title(title, pad=2)

    strip_rgb, lut = color_strip_from_categories(cats_ord)
    ax_strip.imshow(strip_rgb, aspect="auto")
    ax_strip.set_yticks([0]); ax_strip.set_yticklabels(["Category"])
    ax_strip.set_xticks([])


    # heatmap
    vmax = np.nanpercentile(np.abs(M_ord), 95)  # robust symmetric limits
    im = ax_heat.imshow(M_ord, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(np.arange(len(rows))); ax_heat.set_yticklabels(rows)
    ax_heat.set_xticks(np.arange(len(labels_ord)))
    ax_heat.set_xticklabels(labels_ord, rotation=90, fontsize=8)
    # gridlines between rows
    for y in np.arange(len(rows)+1)-0.5:
        ax_heat.axhline(y, color="k", lw=0.3)
    # colorbar
    cax = fig.add_axes([0.98, 0.05, 0.015, (h_heat-0.05)/total_h])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("row-wise z-score of {}".format(metric), rotation=90)

    # legend for categories (compact)
    # (draw off-canvas legend on the right)
    handles = [plt.Line2D([0],[0], marker='s', color=clr, lw=0, markersize=8) for cat, clr in lut.items()]
    labels  = list(lut.keys())
    ax_heat.legend(handles, labels, title="Category", loc="upper left", bbox_to_anchor=(1.02,1.0), fontsize=8)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_clustermap(
    M,                     # 2D array (rows x cols) -- pass RAW effects for display
    rows,                  # list[str], e.g. ["L","R","Delta"]
    cols_meta,             # kept for signature compatibility (not used)
    categories,            # list[str] per column
    col_order,             # list[str] per column (labels)
    title,                 # str
    outpath,               # str path to save
    Z,                     # scipy linkage matrix
    leaves,                # column ordering from dendrogram (list[int])
    metric_label="Novelty bias (PE − CT)",  # colorbar label
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from scipy.cluster.hierarchy import dendrogram

    # ---------- helpers ----------
    def _category_strip(values):
        """Return (1 x N x 3 RGB array for imshow, LUT dict for legend)."""
        vals = pd.Series(values, dtype="object").fillna("Other")
        cats = vals.drop_duplicates().tolist()  # stable first-appearance order
        palette = [
            "#6e40aa","#4277b3","#00a1c1","#1bb182","#6bbb57",
            "#b9c33d","#ffd33d","#fca636","#f17342","#e84e63","#cf3a86"
        ]
        lut = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
        rgb = np.array([to_rgb(lut[c]) for c in vals.tolist()], dtype=float)[None, :, :]
        return rgb, lut

    # ---------- checks & alignment ----------
    M = np.asarray(M, float)
    n_rows, n_cols = M.shape

    if len(col_order) != n_cols:
        raise ValueError(f"len(col_order)={len(col_order)} but M has {n_cols} columns.")
    if len(categories) != n_cols:
        # be forgiving (pad or trim)
        if len(categories) < n_cols:
            categories = list(categories) + ["Other"] * (n_cols - len(categories))
        else:
            categories = list(categories[:n_cols])

    if leaves is None or len(leaves) != n_cols:
        leaves = list(range(n_cols))

    # reorder by dendrogram leaves
    M_ord = M[:, leaves]
    labels_ord = [col_order[i] for i in leaves]
    cats_ord   = [categories[i] for i in leaves]

    # ---------- layout ----------
    h_dend  = 1.2
    h_strip = 0.28
    h_heat  = max(2.2, 0.28 * max(3, n_rows))
    total_h = h_dend + h_strip + h_heat
    fig_w   = max(10, 0.5 * len(labels_ord))

    fig = plt.figure(figsize=(fig_w, total_h), constrained_layout=False)
    ax_dend  = fig.add_axes([0.06, (h_strip + h_heat)/total_h, 0.91, h_dend/total_h])
    ax_strip = fig.add_axes([0.06, (h_heat)/total_h,           0.91, h_strip/total_h])
    ax_heat  = fig.add_axes([0.06, 0.05,                        0.91, (h_heat-0.05)/total_h])

    # ---------- dendrogram ----------
    if Z is not None and len(leaves) == n_cols:
        dendrogram(Z, ax=ax_dend, no_labels=True, color_threshold=0)
    ax_dend.set_xticks([]); ax_dend.set_yticks([])
    ax_dend.set_title(title, pad=2)

    # ---------- category strip (numeric RGB) ----------
    strip_rgb, lut = _category_strip(cats_ord)
    ax_strip.imshow(strip_rgb, aspect="auto")
    ax_strip.set_yticks([0]); ax_strip.set_yticklabels(["Category"])
    ax_strip.set_xticks([])

    # ---------- heatmap (zero-centered diverging) ----------
    finite_vals = M_ord[np.isfinite(M_ord)]
    if finite_vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(np.abs(finite_vals), 95))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    im = ax_heat.imshow(M_ord, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(np.arange(n_rows)); ax_heat.set_yticklabels(list(rows))
    ax_heat.set_xticks(np.arange(len(labels_ord)))
    ax_heat.set_xticklabels(labels_ord, rotation=90, fontsize=8)

    for y in np.arange(n_rows + 1) - 0.5:
        ax_heat.axhline(y, color="k", lw=0.3)

    # colorbar + semantics
    cax = fig.add_axes([0.98, 0.05, 0.015, (h_heat-0.05)/total_h])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(metric_label, rotation=90)
    try:
        cax.text(0.5, 1.02, "Novel ↑ (PE>CT)", ha="center", va="bottom", rotation=90,
                 transform=cax.transAxes, fontsize=8)
        cax.text(0.5, -0.02, "Familiar ↑ (CT>PE)", ha="center", va="top", rotation=90,
                 transform=cax.transAxes, fontsize=8)
    except Exception:
        pass

    # category legend
    handles = [plt.Line2D([0],[0], marker='s', color=clr, lw=0, markersize=8)
               for cat, clr in lut.items()]
    labels  = list(lut.keys())
    if handles:
        ax_heat.legend(handles, labels, title="Category",
                       loc="upper center", bbox_to_anchor=(1.05,1.5), fontsize=8)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

#

def plot_clustermap_dual(
    M_ref, M_other,               # 2D arrays (rows x cols), same rows/cols
    rows,                         # list[str], e.g. ["L","R","Delta"]
    categories_ref,               # list[str] per column (from ref genotype)
    col_labels,                   # list[str] per column
    title,                        # overall title
    title_ref, title_other,       # small titles for each heatmap
    outpath,                      # save path
    Z, leaves,                    # dendrogram & column order
    metric_label="Novelty bias (PE − CT)",
):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Reorder columns by dendrogram leaves
    M1 = M_ref[:, leaves]
    M2 = M_other[:, leaves]
    labels_ord = [col_labels[i] for i in leaves]
    cats_ord   = [categories_ref[i] for i in leaves]

    # Category strip (numeric RGB) + LUT
    strip_rgb, lut = color_strip_from_categories(cats_ord)

    # One symmetric color scale across BOTH panels
    finite_vals = np.concatenate(
        [M1[np.isfinite(M1)], M2[np.isfinite(M2)]]
    ) if (np.isfinite(M1).any() or np.isfinite(M2).any()) else np.array([])
    vmax = float(np.nanpercentile(np.abs(finite_vals), 95)) if finite_vals.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # ----- layout -----
    h_leg   = 0.50        # top legend row
    h_dend  = 1.2
    h_strip = 0.28
    h_h1    = max(2.0, 0.28 * len(rows))
    h_h2    = max(2.0, 0.28 * len(rows))
    h_gap   = 0.08
    total_h = h_leg + h_dend + h_strip + h_h1 + h_gap + h_h2

    fig_w   = max(12, 0.5 * len(labels_ord))
    fig = plt.figure(figsize=(fig_w, total_h), constrained_layout=False)

    # axes (l, b, w, h)
    ax_dend  = fig.add_axes([0.06, (h_strip + h_h1 + h_gap + h_h2)/total_h, 0.91, h_dend/total_h])
    ax_strip = fig.add_axes([0.06, (h_h1 + h_gap + h_h2)/total_h,           0.91, h_strip/total_h])
    ax_h1    = fig.add_axes([0.06, (h_gap + h_h2)/total_h,                  0.91, h_h1/total_h])
    ax_h2    = fig.add_axes([0.06, 0.05,                                     0.91, (h_h2-0.05)/total_h])

    # Top legend (category LUT)
    handles = [plt.Line2D([0],[0], marker='s', color=clr, lw=0, markersize=8)
               for _, clr in lut.items()]
    labels  = list(lut.keys())
    if handles:
        fig.legend(handles, labels, title="Category",
                   loc="upper center", bbox_to_anchor=(0.5, 1.5),
                   ncol=min(len(handles), 6), frameon=False, fontsize=9)

    # Dendrogram
    if Z is not None and len(leaves) == M1.shape[1]:
        dendrogram(Z, ax=ax_dend, no_labels=True, color_threshold=None)
    ax_dend.set_xticks([]); ax_dend.set_yticks([])
    ax_dend.set_title(title, pad=2)

    # Category strip
    ax_strip.imshow(strip_rgb, aspect="auto")
    ax_strip.set_yticks([0]); ax_strip.set_yticklabels(["Category"])
    ax_strip.set_xticks([])

    # Heatmap 1 (ref)
    im1 = ax_h1.imshow(M1, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_h1.set_yticks(np.arange(len(rows))); ax_h1.set_yticklabels(rows)
    ax_h1.set_xticks(np.arange(len(labels_ord))); ax_h1.set_xticklabels(labels_ord, rotation=90, fontsize=8)
    for y in np.arange(len(rows)+1)-0.5: ax_h1.axhline(y, color="k", lw=0.3)
    ax_h1.set_title(title_ref, loc="left", fontsize=11, pad=2)

    # Heatmap 2 (other)
    im2 = ax_h2.imshow(M2, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_h2.set_yticks(np.arange(len(rows))); ax_h2.set_yticklabels(rows)
    ax_h2.set_xticks(np.arange(len(labels_ord))); ax_h2.set_xticklabels(labels_ord, rotation=90, fontsize=8)
    for y in np.arange(len(rows)+1)-0.5: ax_h2.axhline(y, color="k", lw=0.3)
    ax_h2.set_title(title_other, loc="left", fontsize=11, pad=2)

    # One colorbar for both
    cax = fig.add_axes([0.98, 0.05, 0.015, (h_h1 + h_gap + h_h2)/total_h])
    cb = plt.colorbar(im2, cax=cax)
    cb.set_label(metric_label, rotation=90)
    try:
        cax.text(0.5, 1.02, "Novel ↑ (PE>CT)", ha="center", va="bottom", rotation=90,
                 transform=cax.transAxes, fontsize=8)
        cax.text(0.5, -0.02, "Familiar ↑ (CT>PE)", ha="center", va="top", rotation=90,
                 transform=cax.transAxes, fontsize=8)
    except Exception:
        pass

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one(genotype, ref_order=None, ref_Z=None, suffix=""):
    # Load sheets for genotype
    df_L = load_sheet_like(effects_xlsx, f"{genotype}_L")
    df_R = load_sheet_like(effects_xlsx, f"{genotype}_R")
    df_D = load_sheet_like(effects_xlsx, f"{genotype}_Delta")  # optional, will be overridden

    if all(x is None for x in [df_L, df_R, df_D]):
        print(f"[ERROR] No sheets for genotype '{genotype}' in {effects_xlsx}")
        return None

    # ensure category column
    df_L = ensure_category(df_L) if df_L is not None else None
    df_R = ensure_category(df_R) if df_R is not None else None

    # Always recompute Delta with your chosen convention
    df_D = compute_delta_sheet(df_L, df_R, metric=METRIC_NAME, sign=DELTA_FORMULA)

    # Build raw matrix
    M_raw, rows, cols_meta, categories, labels = build_matrix(df_L, df_R, df_D, metric=METRIC_NAME)
    if M_raw.size == 0:
        print(f"[ERROR] Empty matrix for genotype '{genotype}' after build_matrix.")
        return None

    # Order columns
    if ref_order is not None and ref_Z is not None and len(ref_order) == M_raw.shape[1]:
        Z, leaves = ref_Z, ref_order
    else:
        Z, leaves = cluster_columns(zscore_rows(M_raw.copy()))

    # Replace 'Delta' tick label with explicit formula
    rows_for_ticks = [DELTA_LABEL if r == "Delta" else r for r in rows]

    title = f"{genotype} — {METRIC_NAME} (rows: {', '.join(rows_for_ticks)})"
    out = Path(out_png_dir) / f"clustermap_{genotype}_{METRIC_NAME}_novelty_top{top_n}.png"
    plot_clustermap(M_raw, rows_for_ticks, cols_meta, categories, labels, title, str(out), Z, leaves)

    print(f"[OK] Saved {out}")
    return (labels, leaves, Z)

def run_hemi_comparison(geno1="WT", geno2="Shank3", hemi="L"):
    assert hemi in ("L","R"), "hemi must be 'L' or 'R'"

    # Load the per-hemisphere sheets (effect sizes, PE vs CT)
    df1 = load_sheet_like(effects_xlsx, f"{geno1}_{hemi}")
    df2 = load_sheet_like(effects_xlsx, f"{geno2}_{hemi}")
    if df1 is None or df2 is None:
        print(f"[WARN] Missing {hemi} sheet(s) for {geno1}/{geno2}")
        return

    df1 = ensure_category(df1)
    df2 = ensure_category(df2)

    # Align by region_id if available, else acronym
    key = "region_id" if "region_id" in df1.columns and "region_id" in df2.columns else "acronym"
    merged = pd.merge(
        df1[[key, "acronym", "name", "category", METRIC_NAME]],
        df2[[key, METRIC_NAME]],
        on=key, suffixes=(f"_{geno1}", f"_{geno2}")
    )

    # Build a 3-row matrix: WT, Shank3, WT − Shank3
    data = []
    rows = []
    data.append(merged[f"{METRIC_NAME}_{geno1}"].to_numpy(float)); rows.append(f"{geno1} {hemi}")
    data.append(merged[f"{METRIC_NAME}_{geno2}"].to_numpy(float)); rows.append(f"{geno2} {hemi}")
    diff = merged[f"{METRIC_NAME}_{geno1}"] - merged[f"{METRIC_NAME}_{geno2}"]
    data.append(diff.to_numpy(float)); rows.append(f"{geno1}-{geno2} ({hemi})")

    M = np.vstack(data)
    categories = merged["category"].astype(str).tolist()
    labels = merged["acronym"].astype(str).tolist()

    # Cluster columns on pattern across the 3 rows
    Z, leaves = cluster_columns(zscore_rows(M.copy()))

    title = f"{geno1} vs {geno2} — Hemisphere {hemi} ({METRIC_NAME})"
    out = Path(out_png_dir) / f"clustermap_{geno1}_vs_{geno2}_{METRIC_NAME}_{hemi}.png"
    plot_clustermap(
        M, rows, None, categories, labels, title, str(out),
        Z, leaves, metric_label=f"Novelty bias ({METRIC_NAME}: PE − CT)"
    )
    print(f"[OK] {hemi} comparison saved → {out}")

def run_delta_comparison(geno1="WT", geno2="Shank3"):
    # Load Delta (g(Δ)) sheets from your Excel
    df1 = load_sheet_like(effects_xlsx, f"{geno1}_Delta")
    df2 = load_sheet_like(effects_xlsx, f"{geno2}_Delta")
    if df1 is None or df2 is None:
        print("[WARN] Missing Delta sheets for comparison")
        return

    df1 = ensure_category(df1)
    df2 = ensure_category(df2)

    key = "region_id" if "region_id" in df1.columns and "region_id" in df2.columns else "acronym"
    merged = pd.merge(
        df1[[key, "acronym", "name", "category", METRIC_NAME]],
        df2[[key, METRIC_NAME]],
        on=key, suffixes=(f"_{geno1}", f"_{geno2}")
    )

    data = []
    rows = []
    data.append(merged[f"{METRIC_NAME}_{geno1}"].to_numpy(float)); rows.append(f"{geno1} Δ (g(Δ))")
    data.append(merged[f"{METRIC_NAME}_{geno2}"].to_numpy(float)); rows.append(f"{geno2} Δ (g(Δ))")
    diff = merged[f"{METRIC_NAME}_{geno1}"] - merged[f"{METRIC_NAME}_{geno2}"]
    data.append(diff.to_numpy(float)); rows.append(f"{geno1}-{geno2} (Δ)")

    M = np.vstack(data)
    categories = merged["category"].astype(str).tolist()
    labels = merged["acronym"].astype(str).tolist()

    Z, leaves = cluster_columns(zscore_rows(M.copy()))

    title = f"{geno1} vs {geno2} — Hemispheric Δ (g(Δ), {METRIC_NAME})"
    out = Path(out_png_dir) / f"clustermap_{geno1}_vs_{geno2}_{METRIC_NAME}_Delta.png"
    plot_clustermap(
        M, rows, None, categories, labels, title, str(out),
        Z, leaves, metric_label=f"Hemispheric lateralization (g(Δ))"
    )
    print(f"[OK] Δ comparison saved → {out}")


def run_dual(genotype_ref, genotype_other, suffix=""):
    # ----- load sheets -----
    dfL_ref = load_sheet_like(effects_xlsx, f"{genotype_ref}_L")
    dfR_ref = load_sheet_like(effects_xlsx, f"{genotype_ref}_R")
    dfD_ref = load_sheet_like(effects_xlsx, f"{genotype_ref}_Delta")

    dfL_oth = load_sheet_like(effects_xlsx, f"{genotype_other}_L")
    dfR_oth = load_sheet_like(effects_xlsx, f"{genotype_other}_R")
    dfD_oth = load_sheet_like(effects_xlsx, f"{genotype_other}_Delta")

    if all(x is None for x in [dfL_ref, dfR_ref, dfD_ref]):
        print(f"[ERROR] No sheets for reference genotype '{genotype_ref}'"); return
    if all(x is None for x in [dfL_oth, dfR_oth, dfD_oth]):
        print(f"[ERROR] No sheets for other genotype '{genotype_other}'"); return

    # categories
    dfL_ref = ensure_category(dfL_ref) if dfL_ref is not None else None
    dfR_ref = ensure_category(dfR_ref) if dfR_ref is not None else None
    dfL_oth = ensure_category(dfL_oth) if dfL_oth is not None else None
    dfR_oth = ensure_category(dfR_oth) if dfR_oth is not None else None

    # recompute Deltas
    dfD_ref = compute_delta_sheet(dfL_ref, dfR_ref, metric=METRIC_NAME, sign=DELTA_FORMULA)
    dfD_oth = compute_delta_sheet(dfL_oth, dfR_oth, metric=METRIC_NAME, sign=DELTA_FORMULA)

    # build matrices
    M_ref, rows_ref, cols_meta_ref, cats_ref, labels_ref = build_matrix(dfL_ref, dfR_ref, dfD_ref, metric=METRIC_NAME)
    M_oth, rows_oth, cols_meta_oth, cats_oth, labels_oth = build_matrix(dfL_oth, dfR_oth, dfD_oth, metric=METRIC_NAME)

    if M_ref.size == 0 or M_oth.size == 0:
        print("[ERROR] Empty matrix in dual run."); return

    want_rows = [r for r in rows_to_use if r in rows_ref] or rows_ref
    M_ref = align_rows(M_ref, rows_ref, want_rows)
    M_oth = align_rows(M_oth, rows_oth, want_rows)
    M_oth = align_columns(M_oth, labels_oth, labels_ref)

    Z, leaves = cluster_columns(zscore_rows(M_ref.copy()))

    # explicit delta label on both panels
    rows_for_ticks = [DELTA_LABEL if r == "Delta" else r for r in want_rows]

    title_all   = f"{METRIC_NAME}: {genotype_ref} vs {genotype_other} (top {top_n} regions by REF)"
    title_ref   = f"{genotype_ref}"
    title_other = f"{genotype_other}"
    out = Path(out_png_dir) / f"clustermap_{genotype_ref}_vs_{genotype_other}_{METRIC_NAME}_novelty_top{top_n}.png"

    plot_clustermap_dual(
        M_ref, M_oth, rows_for_ticks, cats_ref, labels_ref,
        title_all, title_ref, title_other, str(out), Z, leaves,
        metric_label=f"Novelty bias ({METRIC_NAME}: PE − CT)"
    )
    print(f"[OK] Dual clustermap saved → {out}")

def run_delta_only(genotype, suffix="_delta_only"):
    # Load only the Delta sheet (the one already in your Excel = g(Δ))
    df_D = load_sheet_like(effects_xlsx, f"{genotype}_Delta")
    if df_D is None:
        print(f"[WARN] no Delta sheet for genotype {genotype}")
        return

    df_D = ensure_category(df_D)
    # Build a “matrix” with just one row ("Delta")
    M_raw, rows, cols_meta, categories, labels = build_matrix(
        None, None, df_D, metric=METRIC_NAME
    )

    # Cluster by pattern
    Mz_for_order = zscore_rows(M_raw.copy())
    Z, leaves = cluster_columns(Mz_for_order)

    # Plot raw values (not row z-scored)
    title = f"{genotype} — Hemispheric Δ (g(Δ))"
    out = Path(out_png_dir) / f"clustermap_{genotype}_{METRIC_NAME}_delta_only.png"
    plot_clustermap(M_raw, rows, cols_meta, categories, labels,
                    title, str(out), Z, leaves,
                    metric_label="Hemispheric lateralization (PE − CT)")
    

def main():
    # Usual L+R clustermaps
    #labels, leaves, Z = run_one(ref_genotype)
    #if labels is not None:
    #    run_one(other_geno, ref_order=leaves, ref_Z=Z,
    #            suffix="_ordered_by_"+ref_genotype)

    # Extra: Delta-only clustermaps
    #run_delta_only(ref_genotype)
    #run_delta_only(other_geno)
        # Keep your per-genotype L+R if you like (optional)
    # res = run_one(ref_genotype)
    # if res:
    #     run_one(other_geno, ref_order=res[1], ref_Z=res[2])

    # WT vs Shank3 for L, R, and Δ (each includes a WT−Shank3 row)
    run_hemi_comparison(ref_genotype, other_geno, hemi="L")
    run_hemi_comparison(ref_genotype, other_geno, hemi="R")
    run_delta_comparison(ref_genotype, other_geno)


if __name__ == "__main__":
    main()
