from utils import *
import easygui
import tifffile
from skimage import io
from matplotlib.backends.backend_pdf import PdfPages
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, disk
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


''' [cagedDomains.py] Last Updated: 7/14/2025 by Myles Koppelman '''


def isPointInEllipse(x, y, cx, cy, a, b, angle_deg):

    """
    Check if a point (x, y) lies within an arbitrarily rotated ellipse.

    Parameters
    ----------
    x, y : float
        Coordinates of the point to check.
    cx, cy : float
        Center of the ellipse.
    a, b : float
        Semi-major and semi-minor axes of the ellipse.
    angle_deg : float
        Rotation angle of the ellipse in degrees.

    Returns
    -------
    bool
        True if the point is inside the ellipse, False otherwise.
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = x - cx
    dy = y - cy

    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    return (x_rot / a)**2 + (y_rot / b)**2 <= 1


def findCagedDomains(inner_xlsx_path, outer_xlsx_path):
    """
    Identify pairs of inner and outer domains where the inner domain centroid
    is contained within the boundary of the outer domain.

    Parameters
    ----------
    inner_xlsx_path : str
        Path to the Excel file containing tracked inner domains.
    outer_xlsx_path : str
        Path to the Excel file containing tracked outer domains.

    Returns
    -------
    list of tuples
        List of matched (inner_index, outer_index) pairs.
    """
    inner_xls = pd.ExcelFile(inner_xlsx_path, engine="openpyxl")
    inner_df_all = pd.concat([inner_xls.parse(sheet) for sheet in inner_xls.sheet_names], ignore_index=True)

    outer_xls = pd.ExcelFile(outer_xlsx_path, engine="openpyxl")
    outer_df_all = pd.concat([outer_xls.parse(sheet) for sheet in outer_xls.sheet_names], ignore_index=True)

    caged_domains = []

    for _, outer_row in tqdm(outer_df_all.iterrows(), total=len(outer_df_all), desc="Checking Outer Domains"):
        frame = int(outer_row["Frame"])
        outer_index = int(outer_row["Index"])
        cx = float(outer_row["Centroid_X"])
        cy = float(outer_row["Centroid_Y"])
        maj_axis = float(outer_row["Major_Axis"]) / 2
        min_axis = float(outer_row["Minor_Axis"]) / 2
        orientation = float(outer_row["Orientation"])

        inner_candidates = inner_df_all[inner_df_all["Frame"] == frame]

        for _, inner_row in inner_candidates.iterrows():
            inner_index = int(inner_row["Index"])

            if any(pair[0] == inner_index for pair in caged_domains):
                continue 

            x = float(inner_row["Centroid_X"])
            y = float(inner_row["Centroid_Y"])

            if isPointInEllipse(x, y, cx, cy, maj_axis, min_axis, orientation):
                caged_domains.append((inner_index, outer_index))
                continue 

    return caged_domains


def drawCagedDomains(tif_path, inner_xlsx_path, outer_xlsx_path):
    """
    Generate a new binary TIFF highlighting caged domains and save paired centroid data.

    Parameters
    ----------
    tif_path : str
        Path to the original binary TIFF.
    inner_xlsx_path : str
        Path to Excel file with inner domain tracks.
    outer_xlsx_path : str
        Path to Excel file with outer domain tracks.

    Returns
    -------
    tuple
        Paths to the saved Excel and TIFF files.
    """
    tif_save_path = saveTif("Select path to save new .tif File", "_CAGED", tif_path)
    xlsx_save_path = saveXlsx("Select path to save new .xlsx File", "_CAGED_DATA", tif_path)
    
    tif_stack = io.imread(tif_path).astype(np.uint8)

    inner_xls = pd.ExcelFile(inner_xlsx_path, engine="openpyxl")
    inner_dfs = [inner_xls.parse(sheet) for sheet in inner_xls.sheet_names]
    inner_df_all = pd.concat(inner_dfs, ignore_index=True)
    
    outer_xls = pd.ExcelFile(outer_xlsx_path, engine="openpyxl")
    outer_dfs = [outer_xls.parse(sheet) for sheet in outer_xls.sheet_names]
    outer_df_all = pd.concat(outer_dfs, ignore_index=True)
    

    inner_df1 = inner_dfs[0]
    first_row = inner_df1.iloc[0]
    inner_dc = int(first_row["Domain_Color"])
    outer_dc = 255 if inner_dc == 0 else 0
        
        
    output_stack = np.full_like(tif_stack, fill_value=inner_dc, dtype=np.uint8)

    gap_tolerance_pixels = 3
    
    
    matched_pairs = findCagedDomains(inner_xlsx_path, outer_xlsx_path) 
    matched_outer_indices = set(outer for _, outer in matched_pairs)

    for _, row in outer_df_all.iterrows():
        outer_index = int(row["Index"])
        if outer_index not in matched_outer_indices:
            continue

        frame = int(row["Frame"])
        if not (0 <= frame < tif_stack.shape[0]):
            continue

        minr = int(row["BBox_Y"]) 
        minc = int(row["BBox_X"]) 
        maxr = minr + int(row["BBox_H"]) 
        maxc = minc + int(row["BBox_W"])

        region_slice = tif_stack[frame, minr:maxr, minc:maxc]

        center_r = (maxr - minr) // 2
        center_c = (maxc - minc) // 2

        seed_found = False
        # Check a spiral or square around the center
        max_radius = max(region_slice.shape) // 2
        for radius in range(max_radius):
            for dr, dc in itertools.product(range(-radius, radius+1), repeat=2):
                r = center_r + dr
                c = center_c + dc
                if 0 <= r < region_slice.shape[0] and 0 <= c < region_slice.shape[1]:
                    pixel = region_slice[r, c]
                    if outer_dc == 0 and pixel <= 50:
                        seed = (r, c)
                        seed_found = True
                        break
                    elif outer_dc == 255 and pixel >= 205:
                        seed = (r, c)
                        seed_found = True
                        break
            if seed_found:
                break

        if not seed_found:
            print(f"[Warning] No valid seed found for outer domain index {outer_index} in frame {frame}")
            continue

        connected_mask = flood(region_slice, seed, tolerance=20, connectivity=1)

        if gap_tolerance_pixels > 0:
            selem = disk(gap_tolerance_pixels)
            connected_mask = binary_dilation(connected_mask, footprint=selem)

        if outer_dc == 0:
            connected_mask &= (region_slice <= 50)
        else:
            connected_mask &= (region_slice >= 205)

        output_slice = np.full(region_slice.shape, inner_dc, dtype=np.uint8)
        output_slice[connected_mask] = outer_dc
        output_stack[frame, minr:maxr, minc:maxc] = output_slice


    tifffile.imwrite(tif_save_path, output_stack, photometric='minisblack')
    print(f"Saved matched outer domains to: {tif_save_path}")


    with pd.ExcelWriter(xlsx_save_path, engine='openpyxl', mode='w') as writer:
        for inner, outer in matched_pairs:
            inner_df = inner_df_all[inner_df_all["Index"] == inner].copy()
            outer_df = outer_df_all[outer_df_all["Index"] == outer].copy()

            inner_df = inner_df.rename(columns={
                "Centroid_X": "Inner_Centroid_X",
                "Centroid_Y": "Inner_Centroid_Y"
            })
            outer_df = outer_df.rename(columns={
                "Centroid_X": "Outer_Centroid_X",
                "Centroid_Y": "Outer_Centroid_Y"
            })

            merged_df = pd.merge(inner_df, outer_df, on="Frame", suffixes=("_inner", "_outer"))
            merged_df["Centroid_Distance"] = np.sqrt(
                (merged_df["Inner_Centroid_X"] - merged_df["Outer_Centroid_X"])**2 +
                (merged_df["Inner_Centroid_Y"] - merged_df["Outer_Centroid_Y"])**2
            )

            sheet_name = f"inner{inner}_outer{outer}"
            if len(sheet_name) > 31:
                sheet_name = f"i{inner}_o{outer}"

            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved centroid distance data to: {xlsx_save_path}")

    return xlsx_save_path, tif_save_path 


def calculateEta(xlsx_path, pdf_path, scale):
    """
    Calculate the scaled eta values and fit an exponential decay model to their histogram to calculate mu squared.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file containing merged centroid data.
    pdf_path : str
        Path to save the histogram and exponential fit as a PDF.

    Returns
    -------
    tuple
        Scaled eta values, mean eta, histogram, bin edges, and fit coefficient.
    """

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)
    
    etas = []
    
    for _, row in df_all.iterrows():
        inner_area = float(row["Area_inner"]) 
        outer_area = float(row["Area_outer"])
        rho = float(row["Centroid_Distance"]) * scale
        
        inner_r = np.sqrt(inner_area / np.pi)
        if inner_r < 4:
            continue
        inner_r = inner_r * scale
        outer_r = np.sqrt(outer_area / np.pi) * scale * 0.5
        
        eta = inner_r**2 * rho**2 / outer_r**3
        etas.append(eta)
    
    if not etas:
        print("No valid eta values found.")
        return


    etas = np.array(etas)
    eta_bar = np.mean(etas)
    eta_scaled = etas / eta_bar  # η / η̄

    N = int(np.sqrt(len(eta_scaled)))
    hist, bin_edges = np.histogram(eta_scaled, bins=N)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def exp_decay(x, A):
        return A * np.exp(-x)
    
    popt, _ = curve_fit(exp_decay, bin_centers, hist, p0=[np.max(hist)])
    fit = popt[0]
    
    k_B = 1.380649e-23    
    temp = 298    
    
    mu_squared = (2 * k_B * temp) / (3 * (np.pi**2) * eta_bar)
    
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black', label="Histogram")
        plt.plot(bin_centers, exp_decay(bin_centers, fit), 'r--', label=f"Fit: A·e⁻ˣ, A = {fit:.2f}")
        plt.xlabel("η / η̄")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Scaled η and Exponential Fit, η̄ = {sigfigs(eta_bar, 3)}, mu^2 = {sigfigs(mu_squared, 3)} N")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.show()


    return eta_scaled, eta_bar, hist, bin_edges, fit, mu_squared
    
   
        
def main():
    """
    Main function to execute the caged domain identification and eta calculation workflow.

    Prompts user to select input files, runs domain identification, draws results, calculates eta,
    and saves both graphical and numerical outputs.
    """
    
    print("\n\n\nRunning cagedDomains.py. Please respond to all GUI prompts...\n\n")
    
    _ = easygui.buttonbox(
        msg="cagedDomains.py\n\n    This program allows you to track caged domains and calculate the dipole density (mu^2)\n\n1. Select binary .tif file\n2. Select the tracked .xlsx file for the inner caged domains\n3. Select the tracked .xlsx file for the outer caged domains",
        title="Settings",
        choices=["Continue"]
    )
    
    inner_xlsx_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_DARK_TRACKED.xlsx"
    outer_xlsx_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_LIGHT_TRACKED.xlsx"
    tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm.tif"

    
    tif_path = getTif("Select binary .tif file")
    inner_xlsx_path = getXlsx("Select .xlsx file for the inner caged domains")
    outer_xlsx_path = getXlsx("Select .xlsx file for the outer caged domains")
    
    scale = easygui.multenterbox(
        msg="Enter the number of microns per pixel:",
        title="Settings",
        fields=["Scale (microns/pixel):"],
        values=["0.222222"] 
    )
    
    if scale is None:
        raise Exception("User canceled the input dialog.")

    try:
        scale = float(scale[0]) * 1e-6
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")

    
    xlsx_save_path, _ = drawCagedDomains(tif_path, inner_xlsx_path, outer_xlsx_path)
    
    pdf_path = savePdf("Select location to save .pdf data file", "_DATA", tif_path)
    calculateEta(xlsx_save_path, pdf_path, scale)

    
    

if __name__ == "__main__":
    main()
    

