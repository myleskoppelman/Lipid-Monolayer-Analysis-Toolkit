import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import *
from isolateSingleDomain import *
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

''' [boundaryThickness.py] Last Updated: 7/14/2025 by Myles Koppelman '''


def sigmoid(x, a, b, x0, c):
    """
    Sigmoid function used to model intensity profiles.

    Parameters
    ----------
    x : array_like
        Input x-values (e.g., distance from domain centroid in meters).
    a : float
        Minimum value of the sigmoid curve (baseline).
    b : float
        Height of the sigmoid curve (amplitude).
    x0 : float
        Center of the sigmoid curve (inflection point).
    c : float
        Controls the slope (steepness) of the sigmoid.

    Returns
    -------
    array_like
        Computed sigmoid values for each input x.
    """
    return a + b / (1 + np.exp(-(x - x0) / c))


def analyzeThickness(df, xlsx_path, tif_path, bin_tif_path, r, scale):
    """
    Analyzes the thickness of lipid domain boundaries using intensity profiles
    across the edges of a particle in multiple frames. Fits sigmoid curves to
    estimate boundary sharpness and computes thickness between 10% and 90%
    intensity points.

    Parameters
    ----------
    df : pandas.DataFrame
        Particle data from a single Excel sheet, including centroid positions, frame indices, and domain metadata.
    xlsx_path : str
        Path to the full tracked Excel file.
    tif_path : str
        Path to the original grayscale/raw .tif file.
    bin_tif_path : str
        Path to the binary domain .tif file.
    r : int
        Width (in pixels) of the sampling window for profile extraction.

    Returns
    -------
    dict or None
        Dictionary containing:
            - 'thickness' (float): Estimated domain thickness (meters).
            - 'sigmoid_params' (tuple): Fitted sigmoid parameters (a, b, x0, c).
            - 'x10', 'x90' (float): x-values at 10% and 90% intensities.
            - 'x', 'y' (arrays): Raw profile data.
            - 'x_fit', 'y_fit' (arrays): Fitted sigmoid curve.
        Returns None if fitting fails or usable profiles are not found.
    """
    try:
        idx = int(df.iloc[0]["Index"])
    except (KeyError, IndexError):
        print("Invalid DataFrame: Missing 'Index' column or empty.")
        return None

    if df.get("Holes", pd.Series([False])).any():
        print(f"Skipping sheet 'Particle_{idx}' due to holes.")
        return None

    try:
        img_stack = tifffile.imread(tif_path)
    except Exception as e:
        print(f"Failed to load TIFF file '{tif_path}': {e}")
        return None

    try:
        boundry_img = np.array(isolate(xlsx_path, bin_tif_path, idx))
    except Exception as e:
        print(f"Failed to isolate boundary for Particle_{idx}: {e}")
        return None

    results = []
    start_frame = int(round(df.iloc[0]["Frame"]))

    for _, row in df.iterrows():
        try:
            frame = int(row["Frame"])
            domain_color = int(row["Domain_Color"])

            if frame >= len(img_stack):
                print(f"Frame {frame} exceeds image stack length. Skipping.")
                continue

            img = img_stack[frame]

            boundry_frame = boundry_img[frame - start_frame]
            cx, cy = int(round(row["Centroid_X"])), int(round(row["Centroid_Y"]))

            if not (0 <= cy < boundry_frame.shape[0] and 0 <= cx < boundry_frame.shape[1]):
                print(f"Centroid out of bounds in frame {frame}. Skipping.")
                continue

            hx = np.where(boundry_frame[cy, :] == domain_color)[0]
            vy = np.where(boundry_frame[:, cx] == domain_color)[0]

            if len(hx) < 2 or len(vy) < 2:
                print(f"Insufficient boundary points in frame {frame}. Skipping.")
                continue

            half_r = r // 2
            x_vals = np.arange(-half_r, half_r + 1)

            profile_directions = [('left', -1, hx[0], cy), ('right', +1, hx[-1], cy), ('top', -1, cx, vy[0]), ('bottom', +1, cx, vy[-1])]

            profiles = []
            directions = []

            for label, sign, x0, y0 in profile_directions:
                if label in ('left', 'right'):
                    x_start, x_end = max(0, x0 - half_r), x0 + half_r + 1
                    if x_end - x_start != r:
                        continue
                    profile = img[y0, x_start:x_end].astype(np.float64)
                else:
                    y_start, y_end = max(0, y0 - half_r), y0 + half_r + 1
                    if y_end - y_start != r:
                        continue
                    profile = img[y_start:y_end, x0].astype(np.float64)

                if len(profile) == r:
                    profiles.append(profile)
                    directions.append(sign)

            for profile, sign in zip(profiles, directions):
                for xi, yi in zip(sign * x_vals, profile):
                    results.append((xi, yi))

        except Exception as e:
            print(f"Error processing frame {row.get('Frame', '?')}: {e}")
            continue

    if not results:
        print(f"No usable profiles found for Particle_{idx}.")
        return None

    results = np.array(results)
    x = results[:, 0] * scale
    y = results[:, 1]

    p0 = [np.min(y), np.max(y) - np.min(y), np.median(x), scale]

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0, maxfev=10000)
    except Exception as e:
        print(f"Curve fitting failed for Particle_{idx}: {e}")
        return None

    a, b, x0, c = popt

    try:
        y10 = a + 0.1 * b
        y90 = a + 0.9 * b
        x10 = x0 + c * np.log((b / (y10 - a)) - 1)
        x90 = x0 + c * np.log((b / (y90 - a)) - 1)
        thickness = abs(x90 - x10)
    except Exception as e:
        print(f"Failed to calculate thickness for Particle_{idx}: {e}")
        return None

    x_fit = np.linspace(np.min(x), np.max(x), 500)
    y_fit = sigmoid(x_fit, *popt)

    return {
        "thickness": thickness,
        "sigmoid_params": popt,
        "x10": x10,
        "x90": x90,
        "x": x,
        "y": y,
        "x_fit": x_fit,
        "y_fit": y_fit
    }


def main():
    """
    Main script for analyzing boundary thickness of lipid domains using tracked
    particle data and raw intensity images.

    Workflow:
    ---------
    1. Load tracking Excel file and corresponding raw/binary TIFFs.
    2. Extract particle information for each sheet (domain).
    3. For each particle, extract horizontal/vertical intensity profiles across domain boundaries.
    4. Fit sigmoid to each profile and calculate thickness (x90 - x10).
    5. Save all fit plots to a single PDF.
    6. Save thickness values for each particle to a summary Excel file.

    Notes:
    ------
    - Skips particles with internal holes or missing boundary information.
    - Radius `r` for sampling is determined from the minimum minor axis of domains.
    - Requires external function `isolate()` to extract particle-specific masks.

    Output:
    -------
    - PDF file with sigmoid plots for each domain.
    - Excel file summarizing domain thickness results.
    """
    
    print("\n\n\nRunning boundaryThickness.py. Please respond to all GUI prompts...\n\n\n")
    
    _ = easygui.buttonbox(
        msg="boundaryThickness.py\n\nThis program collects data on the the apparent thickness of a domains boundry due to motion blur and other experimental factors.\n\nOutputs:\n - xxx_THICKESS.xlsx: Data file containing all thickness related data for each domain.\n - xxx_THICKNESS.pdf: File containing intensity plots for each domain\n\n\nPlease respond to all the following prompts for optimal results.",
        title="Settings",
        choices=["Continue"]
    )
    
    # xlsx_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_DARK_TRACKED.xlsx"
    # tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC.tif"
    # bin_tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN.tif"
    
    xlsx_path = getXlsx("Select tracked .xlsx data file")
    tif_path = getTif("Select 8-bit domain .tif file")
    bin_tif_path = getTif("Select binary domain .tif file ")
    
    scale = easygui.multenterbox(
        msg="Enter the number of microns per pixel:",
        title="Settings",
        fields=["Scale (microns/pixel):"],
        values=["0.222222"] # Change these to alter inital settings
    )
    
    if scale is None:
        raise Exception("User canceled the input dialog.")

    try:
        scale = float(scale[0]) * 1e-6
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")
    
    
    data_path = saveXlsx("Select location to save boundary thickness .xlsx data.", "_THICKNESS", tif_path)
    pdf_path = savePdf("Select location to save boundary thickness graph data.", "_THICKNESS", tif_path)

    
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
        if not xls.sheet_names:
            print("No sheets found in Excel file.")
            return
    except Exception as e:
        print(f"Failed to open Excel file: {e}")
        return

    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)

    try:
        r = round(df_all["Minor_Axis"].min() / 1.5)
        r = r + 1 if r % 2 == 0 else r
    except Exception as e:
        print(f"Failed to compute radius from Minor_Axis: {e}")
        return


    all_thickness_data = []

    with PdfPages(pdf_path) as pdf:
        for sheet in tqdm(xls.sheet_names, desc="Processing Particles"):
            try:
                df = xls.parse(sheet)
                result = analyzeThickness(df, xlsx_path, tif_path, bin_tif_path, r, scale)
                if result is None:
                    continue

                all_thickness_data.append({
                    "Sheet": sheet,
                    "Thickness": result["thickness"],
                    "x10": result["x10"],
                    "x90": result["x90"]
                })

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(result["x"], result["y"], s=10, alpha=0.4, label=f"Raw Intensity Profiles of {sheet}")
                ax.plot(result["x_fit"], result["y_fit"], 'r-', label="Sigmoid Fit")
                ax.axvline(result["x10"], color='green', linestyle='--', label="10% (x10)")
                ax.axvline(result["x90"], color='purple', linestyle='--', label="90% (x90)")
                ax.set_title(f"Sigmoid Fit for {sheet}")
                ax.set_xlabel("Meters from Centroid")
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.grid(True)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Error processing sheet {sheet}: {e}")
                continue

    try:
        summary_df = pd.DataFrame(all_thickness_data)
        if summary_df.empty:
            print("No valid thickness data to save.")
            return
        summary_df.to_excel(data_path, index=False)
        print(f"\nSaved results to:\n• {pdf_path}\n• {data_path}")
    except Exception as e:
        print(f"Failed to save summary data: {e}")
        
        
        
        




if __name__ == "__main__":
    main()