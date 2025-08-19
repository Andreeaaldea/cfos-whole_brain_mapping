import brainglobe_heatmap as bgh

import os
import pandas as pd

os.chdir("E:\\Andy\\3dplots")
file_path = 'CBTC.csv'

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
    vmin=-2,  # Minimum value for colormap
    vmax=2,  # Maximum value for colormap
    format="3D",  # 3D format for rendering
    cmap="RdBu"  # Color map for visualization
)

# Display the 3D scene
scene.show()