import pandas as pd
import numpy as np
import tifffile
import os

# 1. Load your updated table from MATLAB, which has 'structure_name' and 'regionCohenD'
os.chdir("Y:\\public\\projects\\AnAl_20240405_Neuromod_PE\\PE_mapping")
file_path = 'CBTC.csv'
actrontable = pd.read_csv("actrontable_with_cohensd.csv")

# 2. Load a region_id mapping (structure_name -> integer ID)
region_map = pd.read_csv("structures.csv")  # e.g. columns: structure_name, region_id

# 3. Merge them by 'structure_name'
merged = pd.merge(
    actrontable,
    region_map,
    on="name",  # Or 'acronym', depending on your data
    how="inner"
)

# 4. Build a dict: region_id -> cohen's d
region_to_d = dict(zip(merged["id"], merged["regionCohenD"]))

# 5. Load annotation_10.tiff (Allen label volume)

label_volume = tifffile.imread("annotation_10.tiff")
print("Label volume shape:", label_volume.shape)

# 6. Prepare an empty array for Cohen's d
cohens_d_volume = np.zeros_like(label_volume, dtype=np.float32)

# 7. Fill in the volume
for r_id, d_val in region_to_d.items():
    mask = (label_volume == r_id)
    cohens_d_volume[mask] = d_val

# Optionally set label=0 (background) to NaN
cohens_d_volume[label_volume == 0] = np.nan

# 8. Save the 3D Cohen’s d volume
tifffile.imwrite(r"Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\figures\cohens_d_3d_regions.tiff", cohens_d_volume)
print("Saved 3D region-based Cohen's d volume: cohens_d_3d_regions.tiff")
