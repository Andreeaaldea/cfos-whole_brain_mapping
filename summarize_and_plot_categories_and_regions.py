import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================== CONFIG ==================
effects_xlsx = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\effect_sizes_PE_vs_CT_by_hemi_with_categories_norm.xlsx"
out_dir      = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\ES_plots_normalized"
top_n        = 160               # top N regions by |metric| to plot
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

def _pick_region_key_cols(df):
    for k in ["region_id", "acronym", "name"]:
        if k in df.columns:
            return k
    return None

def _ensure_region_key(df):
    keycol = _pick_region_key_cols(df)
    if keycol is None:
        df = df.copy()
        df["region_key"] = df.index.astype(str)
        return df, "region_key"
    if keycol != "region_key":
        df = df.copy()
        df["region_key"] = df[keycol].astype(str)
    return df, "region_key"

def build_canonical_order(df_wt, df_sh, primary_metric="g"):
    """
    One stable ordering for a given hemisphere, used across ALL metrics.
    Rank by max(|primary_metric|) across genotypes; tie-break alphabetically by region_key.
    """
    df_wt, key = _ensure_region_key(df_wt)
    df_sh, _   = _ensure_region_key(df_sh)

    # union of regions
    all_keys = sorted(set(df_wt["region_key"]) | set(df_sh["region_key"]))
    wt_vals = df_wt.set_index("region_key")[primary_metric] if primary_metric in df_wt.columns else None
    sh_vals = df_sh.set_index("region_key")[primary_metric] if primary_metric in df_sh.columns else None

    scores = []
    for rk in all_keys:
        vw = abs(float(wt_vals.get(rk))) if wt_vals is not None and pd.notna(wt_vals.get(rk)) else -np.inf
        vs = abs(float(sh_vals.get(rk))) if sh_vals is not None and pd.notna(sh_vals.get(rk)) else -np.inf
        # rank score: max abs value across genotypes
        s = max(vw, vs)
        scores.append((rk, s))

    # sort by score desc, then by region_key asc for determinism
    scores.sort(key=lambda t: (-t[1], t[0]))
    region_order = [rk for rk, _ in scores]
    return region_order


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
from collections import Counter

def build_label_map(df_wt, df_sh):
    """
    Return {region_key -> label} where region_key is the unique key we use
    internally (region_id if available), but the label is the ACRONYM.
    If acronyms collide, disambiguate by appending '(id)'.
    """
    def _one(df):
        # choose key
        if "region_id" in df.columns:
            keys = df["region_id"].astype(str)
        else:
            keys = df.index.astype(str)
        # choose label (prefer acronym, else name, else key)
        if "acronym" in df.columns:
            labs = df["acronym"].astype(str)
        elif "name" in df.columns:
            labs = df["name"].astype(str)
        else:
            labs = keys
        return dict(zip(keys, labs))

    m = {}
    m.update(_one(df_wt))
    m.update(_one(df_sh))

    # disambiguate duplicate labels
    counts = Counter(m.values())
    if any(c > 1 for c in counts.values()):
        # invert: label -> [keys...]
        inv = {}
        for k, lab in m.items(): inv.setdefault(lab, []).append(k)
        for lab, ks in inv.items():
            if len(ks) > 1:
                for k in ks:
                    m[k] = f"{lab} ({k})"  # append id to break ties

    return m

def _pick_region_key_cols(df):
    # prefer unique id for internal alignment
    for k in ["region_id", "acronym", "name"]:
        if k in df.columns:
            return k
    return None

def _ensure_region_key(df):
    keycol = _pick_region_key_cols(df)
    if keycol is None:
        df = df.copy()
        df["region_key"] = df.index.astype(str)
        return df, "region_key"
    if keycol != "region_key":
        df = df.copy()
        df["region_key"] = df[keycol].astype(str)
    return df, "region_key"


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

def build_top_overlap_plot(df_wt, df_sh, metric, hemi, top_n, outpath, region_order, label_map):
    df_wt, _ = _ensure_region_key(df_wt)
    df_sh, _ = _ensure_region_key(df_sh)

    rk_wt = set(df_wt["region_key"]); rk_sh = set(df_sh["region_key"])
    overlap = [rk for rk in region_order if rk in rk_wt and rk in rk_sh]
    if not overlap:
        print(f"[WARN] No overlapping regions between WT and Shank3 for {hemi}."); return
    chosen = overlap[:top_n]

    wt_idx = df_wt.set_index("region_key"); sh_idx = df_sh.set_index("region_key")
    vW = wt_idx.loc[chosen, metric].to_numpy(float); vS = sh_idx.loc[chosen, metric].to_numpy(float)

    lW = wt_idx.loc[chosen, f"{metric}_lo"].to_numpy(float) if f"{metric}_lo" in wt_idx.columns else np.full(len(chosen), np.nan)
    hW = wt_idx.loc[chosen, f"{metric}_hi"].to_numpy(float) if f"{metric}_hi" in wt_idx.columns else np.full(len(chosen), np.nan)
    lS = sh_idx.loc[chosen, f"{metric}_lo"].to_numpy(float) if f"{metric}_lo" in sh_idx.columns else np.full(len(chosen), np.nan)
    hS = sh_idx.loc[chosen, f"{metric}_hi"].to_numpy(float) if f"{metric}_hi" in sh_idx.columns else np.full(len(chosen), np.nan)

    errW = [np.clip(vW - lW, 0, None), np.clip(hW - vW, 0, None)]
    errS = [np.clip(vS - lS, 0, None), np.clip(hS - vS, 0, None)]

    # y-axis: reverse so “top” is at top; labels are acronyms
    y = np.arange(len(chosen))[::-1]
    ylabels = [label_map.get(k, str(k)) for k in chosen][::-1]

    plt.figure(figsize=(11, max(6, len(chosen)*0.35)))
    for i in range(len(y)):
        plt.plot([vW[i], vS[i]], [y[i], y[i]], "-", alpha=0.4)
    plt.errorbar(vW, y, xerr=errW, fmt="o", capsize=3, label="WT")
    plt.errorbar(vS, y, xerr=errS, fmt="o", capsize=3, label="Shank3")
    plt.axvline(0, ls="--")
    plt.yticks(y, ylabels)
    plt.xlabel(f"{metric} (95% CI)")
    plt.title(f"{hemi}: WT vs Shank3 (top {len(chosen)} by fixed order)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()



def plot_region_histograms(df_wt, df_sh, metric, hemi, outpath, region_order=None):
    """
    Plot histograms of metric values for WT and Shank3.
    If region_order is provided, keep bar order consistent across plots.
    """
    # Extract values
    vals_wt = df_wt[[metric, "acronym"]].dropna()
    vals_sh = df_sh[[metric, "acronym"]].dropna()

    if region_order is None:
        # default: sort by |metric| descending
        region_order = vals_wt.set_index("acronym")[metric].abs().sort_values(ascending=False).index.tolist()

    # Align both datasets to region_order
    vals_wt = vals_wt.set_index("acronym").reindex(region_order)
    vals_sh = vals_sh.set_index("acronym").reindex(region_order)

    x = np.arange(len(region_order))
    plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 16})
    width = 0.4

    plt.figure(figsize=(30, max(6, len(region_order)*0.25)))
    plt.bar(x - width/2, vals_wt[metric], width, label="WT", alpha=0.7)
    plt.bar(x + width/2, vals_sh[metric], width, label="Shank3", alpha=0.7)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, region_order, rotation=90)
    plt.ylabel(metric)
    plt.title(f"{hemi}: Histogram of {metric} (WT vs Shank3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    return region_order  # return so same order can be reused
def plot_region_bars(df_wt, df_sh, metric, hemi, outpath, region_order, label_map):
    df_wt, _ = _ensure_region_key(df_wt)
    df_sh, _ = _ensure_region_key(df_sh)

    wt_idx = df_wt.set_index("region_key")
    sh_idx = df_sh.set_index("region_key")

    vals_wt = wt_idx.reindex(region_order)[metric]
    vals_sh = sh_idx.reindex(region_order)[metric]

    x = np.arange(len(region_order))
    width = 0.45
    xlabels = [label_map.get(k, str(k)) for k in region_order]

    plt.figure(figsize=(max(12, len(region_order)*0.18), 6))
    plt.bar(x - width/2, vals_wt.to_numpy(float), width, label="WT", alpha=0.8)
    plt.bar(x + width/2, vals_sh.to_numpy(float), width, label="Shank3", alpha=0.8)
    plt.axhline(0, linewidth=0.8)
    plt.xticks(x, xlabels, rotation=90)
    plt.ylabel(metric)
    plt.title(f"{hemi}: {metric} by region (fixed order)")
    plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()



def plot_distribution_hist(df_wt, df_sh, metric, hemi, outpath):
    """
    True histogram of value distributions (order-free): WT vs Shank3.
    Useful to compare spread/shape; complements the ordered bar figure.
    """
    vW = pd.to_numeric(df_wt[metric], errors="coerce").dropna().to_numpy(float)
    vS = pd.to_numeric(df_sh[metric], errors="coerce").dropna().to_numpy(float)

    plt.figure(figsize=(8,5))
    # identical bins across both groups
    all_vals = np.concatenate([vW, vS]) if len(vW) and len(vS) else (vW if len(vW) else vS)
    bins = 30 if len(all_vals) > 0 else 10
    rng  = (np.nanmin(all_vals), np.nanmax(all_vals)) if len(all_vals) else (-1,1)

    plt.hist(vW, bins=bins, range=rng, alpha=0.6, label="WT", density=False)
    plt.hist(vS, bins=bins, range=rng, alpha=0.6, label="Shank3", density=False)

    plt.axvline(0, ls="--", linewidth=0.8)
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"{hemi}: distribution of {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Choose which metric anchors the ordering for each hemisphere
    anchor_metric = metrics[0] if len(metrics) else "g"

    writer = pd.ExcelWriter(str(Path(out_dir) / "category_summaries.xlsx"))
    for hemi in hemi_panels:
        wt_df, sh_df = load_sheet_like(effects_xlsx, f"WT_{hemi}"), load_sheet_like(effects_xlsx, f"Shank3_{hemi}")
        if wt_df is None or sh_df is None:
            print(f"[WARN] Missing sheets for {hemi}.")
            continue

        wt_df = ensure_category(wt_df)
        sh_df = ensure_category(sh_df)

        # ---- NEW: build a single, stable region order for this hemisphere
        region_order = build_canonical_order(wt_df, sh_df, primary_metric=anchor_metric)
        # Optional sanity check:
        print(f"[{hemi}] first 10 in fixed order: {region_order[:10]}")

        for metric in metrics:
            # Category summaries (unchanged)
            wt_cat = make_category_summary(wt_df, metric, hemi, "WT")
            sh_cat = make_category_summary(sh_df, metric, hemi, "Shank3")
            wt_cat.to_excel(writer, sheet_name=f"{hemi}_WT_by_category_{metric}", index=False)
            sh_cat.to_excel(writer, sheet_name=f"{hemi}_Shank3_by_category_{metric}", index=False)

            # Category comparison plot (unchanged)
            plot_category_comparison(
                wt_cat, sh_cat, metric, hemi,
                outpath=Path(out_dir)/f"category_{hemi}_{metric}_WT_vs_Shank3.png"
            )
            
            label_map = build_label_map(wt_df, sh_df)


            build_top_overlap_plot(
                wt_df, sh_df, metric, hemi, top_n,
                outpath=Path(out_dir)/f"regions_top{top_n}_{hemi}_{metric}_WT_vs_Shank3.png",
                region_order=region_order,
                label_map=label_map
            )

            plot_region_bars(
                wt_df, sh_df, metric, hemi,
                outpath=Path(out_dir)/f"bars_{hemi}_{metric}_WT_vs_Shank3.png",
                region_order=region_order,
                label_map=label_map
            )

            # ---- NEW: true distribution histogram (order-free)
            plot_distribution_hist(
                wt_df, sh_df, metric, hemi,
                outpath=Path(out_dir)/f"hist_{hemi}_{metric}_WT_vs_Shank3.png"
            )

    writer.close()
    print(f"Saved category summaries and plots in: {out_dir}")

if __name__ == "__main__":
    main()
