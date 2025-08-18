import re
import pandas as pd

def wide_to_excel_per_mouse(
    wide_csv_path: str,
    out_xlsx_path: str,
    meta_cols: list[str] | None = None,
    hemisphere_suffixes: tuple[str, str] = ("L", "R"),
) -> None:
    """
    Convert a wide table (meta + many <group>_<mouse>_<L/R> columns) into an Excel
    workbook with one sheet per mouse. Each sheet contains meta columns and the
    two hemisphere columns (if both exist).

    Parameters
    ----------
    wide_csv_path : str
        Path to the input wide CSV (with metadata columns included).
    out_xlsx_path : str
        Path to the output Excel file (.xlsx).
    meta_cols : list[str] | None
        List of metadata columns to carry into every sheet. If None, will auto-detect
        from common meta names found in the input.
    hemisphere_suffixes : tuple[str, str]
        The suffixes that indicate hemisphere columns (default: ("L","R")).
    """
    df = pd.read_csv(wide_csv_path)

    # --- Detect metadata columns if not provided ---
    if meta_cols is None:
        candidates = [
            "region_id", "acronym", "name", "structure_name",
            "structure_id_path", "parent_structure_id", "depth"
        ]
        meta_cols = [c for c in candidates if c in df.columns]
        # Always ensure at least one key column exists in output (fallback)
        if not meta_cols:
            # pick the first non-numeric column as a key
            nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            meta_cols = nonnum[:1] if nonnum else [df.columns[0]]

    # --- Identify numeric value columns (potential mouse hemispheres) ---
    value_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("No numeric mouse columns detected. Check your input wide CSV.")

    # --- Parse mouse columns of the form <base>_<H> where H in {"L","R"} ---
    H = tuple(h.upper() for h in hemisphere_suffixes)
    pattern = re.compile(rf"^(?P<base>.+)_(?P<hemi>{'|'.join(H)})$")
    mouse_map: dict[str, dict[str, str]] = {}  # base -> {"L": colname, "R": colname}

    for col in value_cols:
        m = pattern.match(col)
        if not m:
            # Not a hemisphere column (ignore silently)
            continue
        base = m.group("base")
        hemi = m.group("hemi").upper()
        mouse_map.setdefault(base, {})
        mouse_map[base][hemi] = col

    if not mouse_map:
        raise ValueError(
            "No columns matched the '<group>_<mouse>_<L/R>' pattern. "
            "Check your column names or adjust the hemisphere suffixes."
        )

    # --- Excel sheet name sanitizer (<=31 chars, no :\\/?*[] ) ---
    def sanitize_sheet_name(name: str, used: set[str]) -> str:
        safe = re.sub(r'[:\\/*?\[\]]', "_", name)
        safe = safe[:31] if len(safe) > 31 else safe
        base = safe
        i = 1
        while safe in used:
            suffix = f"_{i}"
            safe = (base[: 31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else (base + suffix)
            i += 1
        used.add(safe)
        return safe

    # --- Write one sheet per mouse base ---
    used_names = set()
    with pd.ExcelWriter(out_xlsx_path, engine="xlsxwriter") as xw:
        for base, lr_cols in sorted(mouse_map.items()):
            # Build the per-mouse frame with meta + L/R where available
            cols = list(meta_cols)
            if "L" in lr_cols:
                cols.append(lr_cols["L"])
            if "R" in lr_cols:
                cols.append(lr_cols["R"])

            # If neither exists (shouldn't happen), skip
            if len(cols) == len(meta_cols):
                continue

            sheet_df = df[cols].copy()

            # Rename the numeric columns to 'L'/'R' for clarity
            rename_map = {}
            if "L" in lr_cols:
                rename_map[lr_cols["L"]] = "L"
            if "R" in lr_cols:
                rename_map[lr_cols["R"]] = "R"
            sheet_df.rename(columns=rename_map, inplace=True)

            # Create a readable sheet name
            sheet_name = sanitize_sheet_name(base, used_names)

            # Write sheet
            sheet_df.to_excel(xw, sheet_name=sheet_name, index=False)

            # Optional: tidy formatting (freeze header row)
            ws = xw.sheets[sheet_name]
            ws.freeze_panes(1, 0)

    print(f"Created Excel with {len(mouse_map)} sheets at: {out_xlsx_path}")
