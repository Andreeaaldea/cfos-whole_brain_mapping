import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================== CONFIG ==================
effects_xlsx = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\effect_sizes_PE_vs_CT_by_hemi_with_categories.xlsx"
out_dir      = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\ES_plots"
top_n        = 114               # top N regions by |metric| to plot
metrics      = ["g", "d"]       # which metrics to plot; choose from {"g","d"}
hemi_panels  = ["L","R","Delta"]  # which hemispheres to include

# If category wasn’t embedded, define a path→category map and we’ll derive it.
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

# Bootstrap settings for category summaries
n_boot = 3000
alpha  = 0.05
rng_seed = 42

# “significant” = CI not crossing 0
def ci_crosses_zero(lo, hi):
    return np.isfinite(lo) and np.isfinite(hi) and (lo <= 0 <= hi)

# ============================================

Path(out_dir).mkdir(parents=True, exist_ok=True)

def assign_category_from_path(path: str) -> str:
    p = "" if pd.isna(path) else str(path)
    for cat, prefix in category_map.items():
        if p.startswith(prefix):
            return cat
    return "Other"

def bootstrap_ci(vals, stat=np.median, n_boot=2000, alpha=0.05, seed=42):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=n, replace=True)) for _ in range(n_boot)])
    est = stat(v)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return est, lo, hi

def load_sheet_like(xlsx, target_prefix):
    """
    Grab the first sheet whose name starts with target_prefix
    (e.g. 'WT_L', 'WT_R', 'WT_Delta', optionally suffixed with _none/_brain_mean).
    """
    xl = pd.ExcelFile(xlsx)
    pat = re.compile(rf"^{re.escape(target_prefix)}($|[_\-])", re.I)
    matches = [s for s in xl.sheet_names if pat.match(s)]
    return pd.read_excel(xlsx, sheet_name=matches[0]) if matches else None

def ensure_category(df):
    if "category" in df.columns:
        return df
    if "structure_id_path" in df.columns:
        df = df.copy()
        df["category"] = df["structure_id_path"].map(assign_category_from_path)
        return df
    df = df.copy()
    df["category"] = "Other"
    return df

def make_category_summary(df, metric, hemi, genotype):
    """
    Returns a per-category summary DataFrame with:
    - n_regions
    - median_abs (|metric|)
    - bootstrap CI for median_abs
    - n_sig and prop_sig (CI not crossing 0)
    """
    # pick CI columns for metric
    lo_col = f"{metric}_lo"
    hi_col = f"{metric}_hi"
    if lo_col not in df.columns or hi_col not in df.columns:
        # If metric doesn't have CI columns (shouldn't happen for g/d), skip counts based on CI
        lo_col = hi_col = None

    rows = []
    for cat, grp in df.groupby("category", dropna=False):
        vals = grp[metric].to_numpy(float)
        est, lo_b, hi_b = bootstrap_ci(np.abs(vals), stat=np.median, n_boot=n_boot, alpha=alpha, seed=rng_seed)
        # significance: CI not crossing 0
        if lo_col and hi_col:
            sig_mask = []
            for _, r in grp.iterrows():
                lo = r.get(lo_col, np.nan)
                hi = r.get(hi_col, np.nan)
                sig_mask.append(not ci_crosses_zero(lo, hi))
            n_sig = int(np.sum(sig_mask))
        else:
            n_sig = np.nan
        rows.append({
            "genotype": genotype,
            "hemi": hemi,
            "category": cat,
            "n_regions": int(len(grp)),
            f"median_abs_{metric}": est,
            f"median_abs_{metric}_lo": lo_b,
            f"median_abs_{metric}_hi": hi_b,
            f"n_sig_{metric}": n_sig,
            f"prop_sig_{metric}": (n_sig / len(grp)) if len(grp) > 0 else np.nan
        })
    out = pd.DataFrame(rows).sort_values(f"median_abs_{metric}", ascending=False)
    return out

def plot_category_comparison(cat_df_wt, cat_df_sh, metric, hemi, outpath):
    """
    Side-by-side points with bootstrap CIs for WT vs Shank3 per category.
    """
    # Align categories
    cats = sorted(set(cat_df_wt["category"]) | set(cat_df_sh["category"]))
    wt_map = cat_df_wt.set_index("category").to_dict("index")
    sh_map = cat_df_sh.set_index("category").to_dict("index")

    y = np.arange(len(cats))
    wt_vals, wt_lo, wt_hi = [], [], []
    sh_vals, sh_lo, sh_hi = [], [], []
    for c in cats:
        rW = wt_map.get(c)
        rS = sh_map.get(c)
        wt_vals.append(rW.get(f"median_abs_{metric}", np.nan) if rW else np.nan)
        wt_lo.append(rW.get(f"median_abs_{metric}_lo", np.nan) if rW else np.nan)
        wt_hi.append(rW.get(f"median_abs_{metric}_hi", np.nan) if rW else np.nan)
        sh_vals.append(rS.get(f"median_abs_{metric}", np.nan) if rS else np.nan)
        sh_lo.append(rS.get(f"median_abs_{metric}_lo", np.nan) if rS else np.nan)
        sh_hi.append(rS.get(f"median_abs_{metric}_hi", np.nan) if rS else np.nan)

    # order by max of the two medians
    order = np.argsort(-(np.nanmax(np.vstack([wt_vals, sh_vals]), axis=0)))
    cats   = [cats[i] for i in order]
    y      = np.arange(len(cats))
    wt_vals= np.array(wt_vals)[order]; wt_lo = np.array(wt_lo)[order]; wt_hi = np.array(wt_hi)[order]
    sh_vals= np.array(sh_vals)[order]; sh_lo = np.array(sh_lo)[order]; sh_hi = np.array(sh_hi)[order]

    # error bar lengths
    wt_err = [np.clip(wt_vals - wt_lo, 0, None), np.clip(wt_hi - wt_vals, 0, None)]
    sh_err = [np.clip(sh_vals - sh_lo, 0, None), np.clip(sh_hi - sh_vals, 0, None)]

    plt.figure(figsize=(10, max(6, len(cats)*0.35)))
    # jitter positions for side-by-side
    offset = 0.15
    plt.errorbar(wt_vals, y - offset, xerr=wt_err, fmt="o", capsize=3, label="WT")
    plt.errorbar(sh_vals, y + offset, xerr=sh_err, fmt="o", capsize=3, label="Shank3")
    plt.axvline(0, ls="--")
    plt.yticks(y, cats)
    plt.xlabel(f"Median |{metric}| (bootstrap 95% CI)")
    plt.title(f"{hemi}: Category comparison (WT vs Shank3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def build_top_overlap_plot(df_wt, df_sh, metric, hemi, top_n, outpath):
    """
    Overlapped region plot: same regions on one axis, WT vs Shank3 points + CIs, connected by a line.
    Only keeps regions present in BOTH tables.
    """
    # Basic meta keys for labeling
    label_cols = [c for c in ["acronym","name","region_id"] if c in df_wt.columns and c in df_sh.columns]
    # intersect regions by region_id if available; else use acronym/name pair
    if "region_id" in label_cols:
        key = "region_id"
        merged = df_wt.merge(df_sh, on=key, suffixes=("_WT","_Sh"))
    else:
        # fallback: join by acronym+name
        keys = [c for c in ["acronym","name"] if c in label_cols]
        merged = df_wt.merge(df_sh, on=keys, suffixes=("_WT","_Sh"))
        key = keys[0] if keys else None

    if merged.empty:
        print(f"[WARN] No overlapping regions between WT and Shank3 for {hemi}.")
        return

    # compute ranking by max absolute metric across genotypes
    merged["_rank_val"] = np.nanmax(
        np.vstack([ np.abs(merged[f"{metric}_WT"].to_numpy(float)),
                    np.abs(merged[f"{metric}_Sh"].to_numpy(float)) ]),
        axis=0
    )
    merged = merged.sort_values("_rank_val", ascending=False).head(top_n)

    # Labels
    def pick_label(r):
        for c in ["acronym","name",key]:
            col = c+"_WT" if c+"_WT" in r else c
            if col in r and pd.notna(r[col]):
                return str(r[col])
        return str(r.name)

    labels = [pick_label(r) for _, r in merged.iterrows()]
    y = np.arange(len(merged))[::-1]  # top at top

    # Values + CI
    vW = merged[f"{metric}_WT"].to_numpy(float)
    lW = merged.get(f"{metric}_lo_WT", pd.Series([np.nan]*len(merged))).to_numpy(float)
    hW = merged.get(f"{metric}_hi_WT", pd.Series([np.nan]*len(merged))).to_numpy(float)
    vS = merged[f"{metric}_Sh"].to_numpy(float)
    lS = merged.get(f"{metric}_lo_Sh", pd.Series([np.nan]*len(merged))).to_numpy(float)
    hS = merged.get(f"{metric}_hi_Sh", pd.Series([np.nan]*len(merged))).to_numpy(float)

    # If CI columns weren’t preserved by merge names, try raw names
    if np.all(~np.isfinite(lW)) and f"{metric}_lo" in df_wt.columns:
        lW = merged[f"{metric}_lo"].to_numpy(float)
        hW = merged[f"{metric}_hi"].to_numpy(float)
    if np.all(~np.isfinite(lS)) and f"{metric}_lo" in df_sh.columns:
        lS = merged[f"{metric}_lo"].to_numpy(float)
        hS = merged[f"{metric}_hi"].to_numpy(float)

    errW = [np.clip(vW - lW, 0, None), np.clip(hW - vW, 0, None)]
    errS = [np.clip(vS - lS, 0, None), np.clip(hS - vS, 0, None)]

    plt.figure(figsize=(11, max(6, len(labels)*0.35)))
    # connecting lines
    for i in range(len(y)):
        plt.plot([vW[i], vS[i]], [y[i], y[i]], "-", alpha=0.4)

    # WT and Sh points with CIs
    plt.errorbar(vW, y, xerr=errW, fmt="o", capsize=3, label="WT")
    plt.errorbar(vS, y, xerr=errS, fmt="o", capsize=3, label="Shank3")

    plt.axvline(0, ls="--")
    plt.yticks(y, labels)
    plt.xlabel(f"{metric} (95% CI)")
    plt.title(f"{hemi}: WT vs Shank3 (top {len(labels)} by |{metric}|)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load all six tables (WT/Shank3 × L/R/Delta)
    def get_pair(hemi):
        wt_df = load_sheet_like(effects_xlsx, f"WT_{hemi}")
        sh_df = load_sheet_like(effects_xlsx, f"Shank3_{hemi}")
        if wt_df is None or sh_df is None:
            print(f"[WARN] Missing sheets for {hemi}.")
            return None, None
        # Ensure category column exists
        wt_df = ensure_category(wt_df)
        sh_df = ensure_category(sh_df)
        return wt_df, sh_df

    # Create an Excel with category summaries
    writer = pd.ExcelWriter(str(Path(out_dir) / "category_summaries.xlsx"))
    for hemi in hemi_panels:
        wt_df, sh_df = get_pair(hemi)
        if wt_df is None: 
            continue

        for metric in metrics:
            # Category summaries
            wt_cat = make_category_summary(wt_df, metric, hemi, "WT")
            sh_cat = make_category_summary(sh_df, metric, hemi, "Shank3")
            sheet_name_wt = f"{hemi}_WT_by_category_{metric}"
            sheet_name_sh = f"{hemi}_Shank3_by_category_{metric}"
            wt_cat.to_excel(writer, sheet_name=sheet_name_wt, index=False)
            sh_cat.to_excel(writer, sheet_name=sheet_name_sh, index=False)

            # Category comparison plot
            plot_category_comparison(
                wt_cat, sh_cat, metric, hemi,
                outpath=Path(out_dir)/f"category_{hemi}_{metric}_WT_vs_Shank3.png"
            )

            # Overlapped top-N region plot
            build_top_overlap_plot(
                wt_df, sh_df, metric, hemi, top_n,
                outpath=Path(out_dir)/f"regions_top{top_n}_{hemi}_{metric}_WT_vs_Shank3.png"
            )

    writer.close()
    print(f"Saved category summaries and plots in: {out_dir}")

if __name__ == "__main__":
    main()
