
import re
import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path

# ---------- CONFIG ----------
collapsed_excel = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\den_collapsed_matrix.xlsx"
sheet_name      = "mean_cells_per_mm3"
out_excel       = r"Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\effect_sizes_PE_vs_CT_by_hemi_with_categories.xlsx"

# Optional per-mouse normalization: "none" or "brain_mean"
normalization   = "none"
mapping_csv     = None   # CSV with columns: mouse_base, genotype (WT/Shank3), condition (PE/CT)

# Category map (prefixes of structure_id_path)
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

# ---------- helpers ----------
def extract_base_and_hemi(col):
    m = re.match(r"^(?P<base>.+)_(?P<hemi>[LR])$", col)
    if not m:
        return None, None
    return m.group("base"), m.group("hemi")

def parse_labels(base_name):
    s = base_name.lower()
    genotype = "WT" if "wt" in s else ("Shank3" if "shank3" in s else None)
    if re.search(r"\bpe\b", s):
        condition = "PE"
    elif re.search(r"\b(ct|ctrl|control)\b", s):
        condition = "CT"
    else:
        toks = re.split(r"[_\-\s]+", s)
        condition = "PE" if "pe" in toks else ("CT" if any(t in {"ct","ctrl","control"} for t in toks) else None)
    return genotype, condition

def assign_category(path: str) -> str:
    if not isinstance(path, str):
        path = "" if pd.isna(path) else str(path)
    for cat, prefix in category_map.items():
        if path.startswith(prefix):
            return cat
    return "Other"

def cohens_d_and_ci(x, y):
    # pooled-SD Cohen's d with ~95% CI
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan, n1, n2,
                np.nan, np.nan, np.nan, np.nan)
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp2 = (((n1-1)*s1**2) + ((n2-1)*s2**2)) / (n1+n2-2)
    sp  = sqrt(sp2) if sp2 > 0 else np.nan
    d   = (m1 - m2) / sp if sp > 0 else np.nan

    # variance of d (Hedges & Olkin) and 95% CI
    var_d = (n1+n2)/(n1*n2) + (d**2)/(2*(n1+n2-2)) if np.isfinite(d) else np.nan
    se_d  = sqrt(var_d) if np.isfinite(var_d) and var_d >= 0 else np.nan
    d_lo  = d - 1.96*se_d if np.isfinite(se_d) else np.nan
    d_hi  = d + 1.96*se_d if np.isfinite(se_d) else np.nan
    return d, d_lo, d_hi, n1, n2, m1, m2, s1, s2

def hedges_g_from_d(d, n1, n2, d_lo, d_hi):
    if not np.isfinite(d) or n1 is None or n2 is None:
        return np.nan, np.nan, np.nan
    J = 1.0 - 3.0 / (4.0*(n1+n2) - 9.0)
    g   = J * d
    g_lo = J * d_lo if np.isfinite(d_lo) else np.nan
    g_hi = J * d_hi if np.isfinite(d_hi) else np.nan
    return g, g_lo, g_hi

def glass_delta(m1, m2, s_control):
    # standardize by control group's SD (CT is control here)
    return (m1 - m2) / s_control if np.isfinite(s_control) and s_control > 0 else np.nan

def d_av(m1, m2, s1, s2):
    # average-SD standardizer (no n weighting)
    denom = np.sqrt((s1**2 + s2**2)/2.0)
    return (m1 - m2) / denom if np.isfinite(denom) and denom > 0 else np.nan

def build_registry(df, mapping_csv=None):
    meta_candidates = ["region_id","acronym","name","structure_id_path","depth","structure_name"]
    meta_cols = [c for c in meta_candidates if c in df.columns]
    value_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in value_cols:
        base, hemi = extract_base_and_hemi(c)
        if base is None:
            continue
        rows.append({"col": c, "base": base, "hemi": hemi})
    reg = pd.DataFrame(rows)
    if mapping_csv is None:
        reg[["genotype","condition"]] = reg["base"].apply(lambda b: pd.Series(parse_labels(b)))
    else:
        mp = pd.read_csv(mapping_csv)
        mp["mouse_base"] = mp["mouse_base"].astype(str)
        reg = reg.merge(mp.rename(columns={"mouse_base":"base"}), on="base", how="left")
    reg = reg[reg["genotype"].isin(["WT","Shank3"]) & reg["condition"].isin(["PE","CT"])]
    return reg, meta_cols, value_cols

def normalize_brain_mean(df, reg):
    df = df.copy()
    for base in sorted(reg["base"].unique()):
        sub = reg[reg["base"] == base]
        cols = []
        if any(sub["hemi"]=="L"):
            cols.append(sub[sub["hemi"]=="L"]["col"].iloc[0])
        if any(sub["hemi"]=="R"):
            cols.append(sub[sub["hemi"]=="R"]["col"].iloc[0])
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        denom = pd.concat([df[c] for c in cols], axis=1).mean(axis=1)
        scale = denom.mean(skipna=True)
        if np.isfinite(scale) and scale != 0:
            for c in cols:
                df[c] = df[c] / scale
    return df

# ---------- load + optional normalization ----------
df = pd.read_excel(collapsed_excel, sheet_name=sheet_name)
reg, meta_cols, value_cols = build_registry(df, mapping_csv)
if normalization == "brain_mean":
    df = normalize_brain_mean(df, reg)

# ---------- effect tables ----------
def effect_table_for(genotype: str, hemi: str) -> pd.DataFrame:
    sub = reg[(reg["genotype"] == genotype) & (reg["hemi"] == hemi)]
    pe_cols = sub[sub["condition"] == "PE"]["col"].tolist()
    ct_cols = sub[sub["condition"] == "CT"]["col"].tolist()
    rows = []

    for i in range(len(df)):
        meta = {k: df.at[i,k] for k in meta_cols if k in df.columns}
        # pick category from structure_id_path if present
        if "structure_id_path" in df.columns:
            meta["category"] = assign_category(str(df.at[i,"structure_id_path"]))
        else:
            meta["category"] = "Other"

        x = df.loc[i, pe_cols].to_numpy(dtype=float) if pe_cols else np.array([])
        y = df.loc[i, ct_cols].to_numpy(dtype=float) if ct_cols else np.array([])

        d, d_lo, d_hi, n1, n2, m1, m2, s1, s2 = cohens_d_and_ci(x, y)
        g, g_lo, g_hi = hedges_g_from_d(d, n1, n2, d_lo, d_hi)
        # Glass's Δ uses CT SD (y)
        glass = glass_delta(m1, m2, s2)
        dav   = d_av(m1, m2, s1, s2)

        # variance diagnostics
        sd_ratio = np.nan
        var_ratio = np.nan
        if np.isfinite(s1) and np.isfinite(s2) and min(s1, s2) > 0:
            sd_ratio = max(s1, s2) / min(s1, s2)
            var_ratio = sd_ratio**2
        var_flag = (sd_ratio >= 2.0) if np.isfinite(sd_ratio) else False  # flag if SDs differ >= 2x

        rows.append({
            **meta,
            "g": g, "g_lo": g_lo, "g_hi": g_hi,
            "d": d, "d_lo": d_lo, "d_hi": d_hi,
            "glass_delta": glass,
            "d_av": dav,
            "n_PE": n1, "n_CT": n2,
            "mean_PE": m1, "mean_CT": m2,
            "sd_PE": s1, "sd_CT": s2,
            "sd_ratio": sd_ratio, "var_ratio": var_ratio, "var_flag_SDx2": var_flag
        })
    return pd.DataFrame(rows)

def effect_table_delta_for(genotype: str) -> pd.DataFrame:
    # require paired L/R per base
    pivot = (reg[reg["genotype"]==genotype]
             .pivot_table(index=["base","condition"], columns="hemi", values="col", aggfunc="first")
             .reset_index())
    pe = pivot[pivot["condition"]=="PE"].dropna(subset=["L","R"])
    ct = pivot[pivot["condition"]=="CT"].dropna(subset=["L","R"])

    # build per-region per-mouse deltas
    deltas_pe = [ (df[r["R"]] - df[r["L"]]).to_numpy(dtype=float) for _, r in pe.iterrows() ]
    deltas_ct = [ (df[r["R"]] - df[r["L"]]).to_numpy(dtype=float) for _, r in ct.iterrows() ]

    rows = []
    for i in range(len(df)):
        meta = {k: df.at[i,k] for k in meta_cols if k in df.columns}
        if "structure_id_path" in df.columns:
            meta["category"] = assign_category(str(df.at[i,"structure_id_path"]))
        else:
            meta["category"] = "Other"

        x = np.array([col[i] for col in deltas_pe], float) if deltas_pe else np.array([])
        y = np.array([col[i] for col in deltas_ct], float) if deltas_ct else np.array([])

        d, d_lo, d_hi, n1, n2, m1, m2, s1, s2 = cohens_d_and_ci(x, y)
        g, g_lo, g_hi = hedges_g_from_d(d, n1, n2, d_lo, d_hi)
        # Glass's Δ on deltas: standardize by CT delta SD
        glass = glass_delta(m1, m2, s2)
        dav   = d_av(m1, m2, s1, s2)

        sd_ratio = np.nan
        var_ratio = np.nan
        if np.isfinite(s1) and np.isfinite(s2) and min(s1, s2) > 0:
            sd_ratio = max(s1, s2) / min(s1, s2)
            var_ratio = sd_ratio**2
        var_flag = (sd_ratio >= 2.0) if np.isfinite(sd_ratio) else False

        rows.append({
            **meta,
            "g": g, "g_lo": g_lo, "g_hi": g_hi,
            "d": d, "d_lo": d_lo, "d_hi": d_hi,
            "glass_delta": glass,
            "d_av": dav,
            "n_PE": n1, "n_CT": n2,
            "mean_PE_delta": m1, "mean_CT_delta": m2,
            "sd_PE_delta": s1, "sd_CT_delta": s2,
            "sd_ratio": sd_ratio, "var_ratio": var_ratio, "var_flag_SDx2": var_flag
        })
    return pd.DataFrame(rows)

# ---------- compute and save ----------
wt_L  = effect_table_for("WT","L")
wt_R  = effect_table_for("WT","R")
wt_D  = effect_table_delta_for("WT")
sh_L  = effect_table_for("Shank3","L")
sh_R  = effect_table_for("Shank3","R")
sh_D  = effect_table_delta_for("Shank3")

with pd.ExcelWriter(out_excel) as xw:
    wt_L.to_excel(xw, sheet_name=f"WT_L_{normalization}", index=False)
    wt_R.to_excel(xw, sheet_name=f"WT_R_{normalization}", index=False)
    wt_D.to_excel(xw, sheet_name=f"WT_Delta_{normalization}", index=False)

    sh_L.to_excel(xw, sheet_name=f"Shank3_L_{normalization}", index=False)
    sh_R.to_excel(xw, sheet_name=f"Shank3_R_{normalization}", index=False)
    sh_D.to_excel(xw, sheet_name=f"Shank3_Delta_{normalization}", index=False)

print(f"Wrote: {out_excel}")
