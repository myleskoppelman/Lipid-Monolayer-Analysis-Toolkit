import tifffile
from skimage import io
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, disk, binary_erosion
from scipy.ndimage import binary_fill_holes
import easygui
import os
import pandas as pd
import numpy as np
from utils import *

''' [isolateDomains.py] Last Updated: 6/6/2025 by Myles Koppelman '''


def fillHoles(frame, domain_color):
    domain_mask = frame == domain_color

    if not np.any(domain_mask):
        return frame

    filled_mask = binary_fill_holes(domain_mask)

    new_pixels = filled_mask & ~domain_mask
    frame[new_pixels] = domain_color
    
    return frame

    
    


def isolate(data_path, tif_path):
    """
    Draws domains for the specified particles in a multi-page TIFF stack by performing a flood-fill segmentation
    for each region (domain) specified in the provided Excel file.

    Parameters:
    ----------
    data_path : str
        Path to the Excel file containing region properties for particles. It must have sheets with columns
        'Frame', 'BBox_X', 'BBox_Y', 'BBox_W', 'BBox_H', 'Centroid_X', and 'Centroid_Y' for each region.
    tif_path : str
        Path to the multi-page TIFF file containing the image stack of interest.

    Notes:
    -----
    - The function prompts the user to save the output TIFF file.
    - For each region listed in the Excel file:
        - It extracts a small bounding box from the corresponding frame in the TIFF stack.
        - Performs flood-fill starting from the centroid with a specified tolerance (20 here).
        - Optionally applies binary dilation to close small gaps (default: 3 pixels).
        - Fills the output stack with the segmented domain, leaving the rest as white (255).
    - The output TIFF stack (`_ISO.tif`) highlights the segmented domains.
    - The final result is saved and the user is notified of the file location.
    """
    
        
    # ----- Open input files
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)
    tif_stack = io.imread(tif_path).astype(np.uint8)
    


    df1 = dfs[0]
    first_row = df1.iloc[0]
    domain_color = int(first_row["Domain_Color"])


    
    if domain_color == 0:
        background_color = 255
    else:
        background_color = 0
        

        
    output_stack = np.full_like(tif_stack, fill_value=background_color, dtype=np.uint8)
    

    # ----- Gap Tolerance. This program draws the desired domains from the centroid outwards, considering only similar colored pixels
    # around it. The gap tolerance allows you to modify the amount of different colored pixels it can see before stopping drawing
    gap_tolerance_pixels = 1

    for _, row in df_all.iterrows():

        frame = int(row["Frame"])
        minr = int(row["BBox_Y"]) - 2
        minc = int(row["BBox_X"]) - 2
        maxr = minr + int(row["BBox_H"]) + 4
        maxc = minc + int(row["BBox_W"]) + 4

        if 0 <= frame < tif_stack.shape[0]:
            try:
                region_slice = tif_stack[frame, minr:maxr, minc:maxc]
            except IndexError:
                region_slice = tif_stack[minr:maxr, minc:maxc]
            
            fillHoles(region_slice, domain_color)

            centroid_r = int(row["Centroid_Y"]) - minr
            centroid_c = int(row["Centroid_X"]) - minc

            if 0 <= centroid_r < region_slice.shape[0] and 0 <= centroid_c < region_slice.shape[1]:
                connected_mask = flood(region_slice, (centroid_r, centroid_c), tolerance=20, connectivity=1)
                if gap_tolerance_pixels > 0:
                    selem = disk(gap_tolerance_pixels)
                    connected_mask = binary_dilation(connected_mask, footprint=selem)

                    # Only keep pixels similar to domain color (black or white, within some threshold)
                    if domain_color == 0:
                        connected_mask &= (region_slice <= 50)
                    else:
                        connected_mask &= (region_slice >= 205)  
                perimeter_mask = connected_mask

                # Create output slice with background color
                output_slice = np.full(region_slice.shape, background_color, dtype=np.uint8)

                # Draw only perimeter in domain color
                output_slice[perimeter_mask] = domain_color
                try:
                    target_region = output_stack[frame, minr:maxr, minc:maxc]
                except IndexError:
                    target_region = output_stack[minr:maxr, minc:maxc]

                # Create mask of where output_slice wants to write
                write_mask = (output_slice == domain_color)

                # Only overwrite if the target pixel is not already domain_color
                safe_write_mask = write_mask & (target_region != domain_color)
                target_region[safe_write_mask] = domain_color

                try:
                    output_stack[frame, minr:maxr, minc:maxc] = target_region
                except IndexError:
                    output_stack[minr:maxr, minc:maxc] = target_region

    return output_stack




def isolateDomains(data_path, tif_path):
    save_path = saveTif("Save Output .tif File", "_DOMAINS", tif_path)
    output_stack = isolate(data_path, tif_path)
    
    tifffile.imwrite(save_path, output_stack, photometric='minisblack')
    print(f"Saved Domains to: {save_path}")
    
    return save_path

    
    
    


def main():
    data_path = getXlsx("Select Particle Tracking Data")
    tif_path = getTif("Select Binary .tif File")

    isolateDomains(data_path, tif_path)
    
    
if __name__ == "__main__":
    main()
    
