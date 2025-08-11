
import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
from compute_dff_folder import compute_dff_folder
from tqdm import tqdm
import fnmatch

def process_single_brain(args):
    input_folder, output_folder = args
    compute_dff_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        n_components=2,
        sample_fraction=0.05,
        denoise=True,
        background_radius=20,
        median_size=2,
        threshold=0.1
    )

def find_brain_folders(base_input_dir, search_suffix="stitchedImages_100/2"):
    matches = []
    for root, dirs, files in os.walk(base_input_dir):
        if fnmatch.fnmatch(root, f"*{search_suffix}"):
            matches.append(os.path.normpath(root))
    return matches

def batch_process_recursive(input_root, output_root, search_suffix="stitchedImages_100/2", dry_run=False):
    brain_folders = find_brain_folders(input_root, search_suffix)
    print(f"Found {len(brain_folders)} matching folders.")

    if len(brain_folders) == 0:
        print("No matching folders found. Check your --input_root or folder structure.")
        return

    if dry_run:
        print("🟡 Dry run mode: listing matched folders without processing:")
        for f in brain_folders:
            print(f"  - {f}")
        return

    args_list = []
    for in_path in brain_folders:
        rel_path = os.path.relpath(in_path, input_root)
        out_path = os.path.join(output_root, rel_path)
        os.makedirs(out_path, exist_ok=True)
        args_list.append((in_path, out_path))
    num_cores = min(16, cpu_count())  
    with Pool(num_cores) as pool:
        pool.map(process_single_brain, args_list)

    print("Recursive batch processing complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Root of tree with subfolders")
    parser.add_argument("--output_root", required=True, help="Output base directory")
    parser.add_argument("--search_suffix", default="stitchedImages_100/2", help="Relative folder pattern to match")
    parser.add_argument("--dry_run", action="store_true", help="Only list matching folders without processing")

    args = parser.parse_args()

    batch_process_recursive(args.input_root, args.output_root, args.search_suffix, args.dry_run)
