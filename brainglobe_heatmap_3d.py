from pathlib import Path
from typing import Tuple, Union, List
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile
from brainglobe_utils.general.system import ensure_directory_exists
import tifffile


def cohens_d_heatmap_from_points(
    group1_points_list: List[np.ndarray],
    group2_points_list: List[np.ndarray],
    image_resolution: float,
    image_shape: Tuple[int, int, int],
    region_mask: np.ndarray,
    smoothing: Union[float, None] = None,
    output_filename: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Generate a Cohen's d heatmap based on the density of cells in each region.
    
    For each experimental group (list of animal cell coordinate arrays),
    the function computes the density (cell count / region volume) for each region
    defined in region_mask. Cohen's d is then calculated between group1 and group2
    for every region. The resulting heatmap is an array of the same shape as region_mask,
    where every voxel is assigned the Cohen's d value of its region.
    
    Parameters
    ----------
    group1_points_list : list of np.ndarray
        A list (length = number of animals in group1) of cell coordinate arrays.
        Each array should have shape (n_points, 3) in voxel coordinates.
    group2_points_list : list of np.ndarray
        A list (length = number of animals in group2) of cell coordinate arrays.
    image_resolution : float
        The resolution of the image (assumed isotropic; e.g. in microns).
    image_shape : Tuple[int, int, int]
        The shape of the 3D volume (e.g. as defined by your atlas or downsampled image).
    region_mask : np.ndarray
        A labeled image with the same shape as image_shape. Each voxel’s value
        should correspond to a brain region (with 0 reserved for background if desired).
    smoothing : float or None, optional
        Smoothing factor in the same physical units as image_resolution.
        If provided, it is converted to voxel units and applied as a Gaussian filter.
    output_filename : str or Path or None, optional
        If provided, the resulting heatmap is saved as a TIFF file.
    
    Returns
    -------
    np.ndarray
        A 3D array of the Cohen's d heatmap.
    """
    # Determine the unique region labels (excluding background label 0)
    region_labels = np.unique(region_mask)
    region_labels = region_labels[region_labels != 0]
    
    # Prepare dictionaries to store per-animal densities for each region.
    group1_densities = {label: [] for label in region_labels}
    group2_densities = {label: [] for label in region_labels}
    
    def get_density_for_animal(points: np.ndarray) -> dict:
        """
        For a single animal's cell coordinates, count the number of cells
        falling in each region and normalize by region volume.
        """
        # Ensure coordinates are integers (assumes points are in voxel space)
        points_int = np.round(points).astype(int)
        # Initialize counts for each region
        counts = {label: 0 for label in region_labels}
        # Loop over points (a vectorized approach could be implemented if needed)
        for pt in points_int:
            # Check that the point is within the image bounds
            if np.all(pt >= 0) and np.all(pt < np.array(image_shape)):
                label = region_mask[tuple(pt)]
                if label in counts:
                    counts[label] += 1
        # Normalize counts by the volume (number of voxels) for each region
        densities = {}
        for label in region_labels:
            region_volume = np.sum(region_mask == label)
            densities[label] = counts[label] / region_volume if region_volume > 0 else 0
        print("Densities computed for this animal:", densities)
        return densities
    
    # Process each animal in group 1
    for points in group1_points_list:
        densities = get_density_for_animal(points)
        for label in region_labels:
            group1_densities[label].append(densities[label])
    
    # Process each animal in group 2
    for points in group2_points_list:
        densities = get_density_for_animal(points)
        for label in region_labels:
            group2_densities[label].append(densities[label])
    
    # Compute Cohen's d for each region
    cohens_d_values = {}
    for label in region_labels:
        data1 = np.array(group1_densities[label])
        data2 = np.array(group2_densities[label])
        mean1 = data1.mean()
        mean2 = data2.mean()
        std1 = data1.std(ddof=1)
        std2 = data2.std(ddof=1)
        n1 = len(data1)
        n2 = len(data2)
        # Calculate the pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        # Handle the case where pooled_std is zero
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        cohens_d_values[label] = cohens_d
    
    # Create an output heatmap: assign each voxel the Cohen's d value for its region
    heatmap_array_chd = np.zeros(image_shape, dtype=np.float32)
    for label, d_val in cohens_d_values.items():
        heatmap_array_chd[region_mask == label] = d_val

    # Create an output heatmap: group 1 density 
    heatmap_array_mg1 = np.zeros(image_shape, dtype=np.float32)
    for label, d_val in cohens_d_values.items():
        heatmap_array_mg1[region_mask == label] = d_val

    # Create an output heatmap: group 1 density 
    heatmap_array_mg2 = np.zeros(image_shape, dtype=np.float32)
    for label, d_val in cohens_d_values.items():
        heatmap_array_mg2[region_mask == label] = d_val
    
    # Optionally apply smoothing (convert smoothing factor to voxel units)
    if smoothing is not None:
        sigma_voxels = smoothing / image_resolution
        heatmap_array_chd = gaussian_filter(heatmap_array_chd, sigma=sigma_voxels)
    
    # Save the heatmap as a TIFF if an output filename is provided.
    if output_filename is not None:
        ensure_directory_exists(Path(output_filename).parent)
        tifffile.imwrite(str(output_filename), heatmap_array_chd.astype(np.float32))
    
    return heatmap_array_chd
