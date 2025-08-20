import brainglobe_heatmap as bgh
import os
import pandas as pd
# Example CSV file path
os.chdir("Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data")
file_path = 'effect_sizes_PE_vs_CT_by_hemi_with_categories.xlsx'
dd = pd.read_excel(file_path, sheet_name='WT_L_none', usecols=['acronym', 'g'])
df = pd.DataFrame(dd)
dict_without_headers = dict(zip(df["acronym"], df["g"]))
val_only = df.values
all_areas = df.to_dict()

# Read specific columns (e.g., 'Column1' and 'Column3')
columns_to_import = ['Acronym', 'dNormcohen_d']
dd = pd.read_csv(file_path, usecols=columns_to_import)
df = pd.DataFrame(dd)
dict_without_headers = dict(zip(df["Acronym"], df["dNormcohen_d"]))

# Create the 3D scene with brainrender
scene = bgh.Heatmap(
    dict_without_headers,
    position=(8000, 5000, 5000),  # Adjust these values for position as needed
    orientation="sagittal",  # Change to 'sagittal', 'horizontal', or a tuple (x,y,z)
    thickness=500,  # Thickness of the section
    title="3D Heatmap - Frontal Orientation",
    vmin=-1.75,  # Minimum value for colormap
    vmax=1.75,  # Maximum value for colormap
    format="3D",  # 3D format for rendering
    cmap="coolwarm"  # Color map for visualization
)

# Display the 3D scene
scene.show()