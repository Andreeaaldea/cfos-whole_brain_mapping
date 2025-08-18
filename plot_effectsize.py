import argparse, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def pick_label(row):
    for k in ("acronym", "name", "region_id"):
        if k in row and pd.notna(row[k]):
            return str(row[k])
    return str(row.name)

def top_table(df, metric, top_n):
    # choose rows with finite metric, sort by |metric|
    s = df[metric].astype(float)
    mask = np.isfinite(s)
    out = df.loc[mask].copy()
    out["_abs"] = out[metric].abs()
    out = out.sort_values("_abs", ascending=False).head(top_n).drop(columns=["_abs"])
    return out

def plot_top_horizontal(df, metric, lo_col, hi_col, title, outpath):
    if df.empty:
        print(f"[WARN] nothing to plot for {title}")
        return
    # Build labels and values
    labels = df.apply(pick_label, axis=1).tolist()
    vals   = df[metric].to_numpy(float)
    lo     = df[lo_col].to_numpy(float)
    hi     = df[hi_col].to_numpy(float)

    # order top->bottom visually (largest at top)
    order = np.argsort(-np.abs(vals))
    labels = [labels[i] for i in order]
    vals   = vals[order]
    lo     = lo[order]
    hi     = hi[order]

    y = np.arange(len(vals))
    # error bars from point to CI bounds
    err_low  = np.clip(vals - lo, 0, None)
    err_high = np.clip(hi - vals, 0, None)

    plt.figure(figsize=(10, max(6, len(vals)*0.3)))
    plt.errorbar(vals, y, xerr=[err_low, err_high], fmt="o", capsize=3)
    plt.axvline(0, linestyle="--")
    plt.yticks(y, labels)
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved plot: {outpath}")

def process_sheet(df, sheet_name, outdir, top_n):
    # Expect columns: g, g_lo, g_hi, d, d_lo, d_hi plus meta (acronym/name/region_id)
    keep_cols = [c for c in ["region_id","acronym","name","structure_id_path","depth",
                             "g","g_lo","g_hi","d","d_lo","d_hi",
                             "n_PE","n_CT","mean_PE","mean_CT","sd_PE","sd_CT",
                             "mean_PE_delta","mean_CT_delta","sd_PE_delta","sd_CT_delta"]
                 if c in df.columns]
    df = df[keep_cols].copy()

    # Top-40 by |g| and |d|
    top_g = top_table(df, "g", top_n)
    top_d = top_table(df, "d", top_n)

    # Save the ranked tables too
    csv_g = Path(outdir)/f"top{top_n}_{sheet_name}_by_g.csv"
    csv_d = Path(outdir)/f"top{top_n}_{sheet_name}_by_d.csv"
    top_g.to_csv(csv_g, index=False)
    top_d.to_csv(csv_d, index=False)

    # Plots
    plot_top_horizontal(top_g, "g", "g_lo", "g_hi",
                        f"{sheet_name}: Top {top_n} by |g| (Hedges' g, 95% CI)",
                        Path(outdir)/f"top{top_n}_{sheet_name}_g.png")
    plot_top_horizontal(top_d, "d", "d_lo", "d_hi",
                        f"{sheet_name}: Top {top_n} by |d| (Cohen's d, 95% CI)",
                        Path(outdir)/f"top{top_n}_{sheet_name}_d.png")

def main(effects_xlsx, outdir, top_n):
    sheets = ["WT_L","WT_R","WT_Delta","Shank3_L","Shank3_R","Shank3_Delta"]
    xls = pd.ExcelFile(effects_xlsx)
    for sh in sheets:
        if sh not in xls.sheet_names:
            print(f"[WARN] sheet missing: {sh}")
            continue
        df = pd.read_excel(effects_xlsx, sheet_name=sh)
        process_sheet(df, sh, outdir, top_n)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("effects_xlsx", help="Excel with effect size sheets (from previous step).")
    p.add_argument("outdir", help="Folder to write plots and CSVs.")
    p.add_argument("--top", type=int, default=40, help="Number of top regions to show (default 40).")
    args = p.parse_args()
    main(args.effects_xlsx, args.outdir, args.top)
