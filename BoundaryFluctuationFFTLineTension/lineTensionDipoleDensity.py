import os
from skimage import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import linregress
from skimage.segmentation import flood
from skimage.morphology import binary_erosion
import math
from utils import *
from isolateAllDomains import *
from dipoleDensity import * 


''' [isolateDomains.py] Last Updated: 6/6/2025 by Myles Koppelman 

    Adapted from MATLAB code by Andrew Nyugen Circa 2007 '''



def calculatePerimeter(tif_path, df):
    """
    Extracts the boundary pixel coordinates of tracked domains from a binary .tif stack.

    Args:
        tif_path (str): Path to the binary .tif file.
        df (pd.DataFrame): DataFrame containing domain tracking information.

    Returns:
        list of np.ndarray: List where each entry is an array of (y, x) coordinates (centered on domain centroid) for each frame.
    """
    tif_stack = io.imread(tif_path).astype(np.uint8)
    
    coords = [] # Array to store coords for every frame of domain
        
    for _, row in df.iterrows(): # loop over each frame domain is present
        
        frame = int(row["Frame"])
        
        if not (0 <= frame < len(tif_stack)):
            continue
        
        domain_color = int(row["Domain_Color"])
        centroid_y = int(row["Centroid_Y"])
        centroid_x = int(row["Centroid_X"])
        bbx = int(row["BBox_X"]) 
        bby = int(row["BBox_Y"]) 
        bbw = int(row["BBox_W"]) 
        bbh = int(row["BBox_H"])
        
 
        padding = 5
        try:
            img_height, img_width = tif_stack[frame].shape
        except:
            img_height, img_width = tif_stack.shape

        x_start = max(bbx - padding, 0)
        y_start = max(bby - padding, 0)
        x_end = min(bbx + bbw + padding, img_width)
        y_end = min(bby + bbh + padding, img_height)

        try:
            frame_slice = tif_stack[frame][y_start:y_end, x_start:x_end]
        except:
            frame_slice = tif_stack[y_start:y_end, x_start:x_end]

        domain_mask = (frame_slice == domain_color)

        if not np.any(domain_mask):
            continue  

        yy, xx = np.where(domain_mask)
        centroid_y = int(np.round(np.mean(yy)))
        centroid_x = int(np.round(np.mean(xx)))


        connected_mask = flood(domain_mask, (centroid_y, centroid_x), tolerance=0)
        
        if np.array_equal(connected_mask, domain_mask):
            # The flood filled the entire domain_mask, so skip this case
            continue
        
        perimeter_mask = connected_mask & ~binary_erosion(connected_mask)

        py, px = np.where(perimeter_mask)
        domain_perimeter = np.column_stack((py - centroid_y, px - centroid_x))

        coords.append(domain_perimeter)


    return coords



def fft(coords, num_harmonics=25):
    """
    Computes Fourier coefficients and power spectra for a series of boundary coordinates.

    Args:
        coords (list of np.ndarray): Boundary coordinates for each frame.
        num_harmonics (int): Number of harmonics to calculate.

    Returns:
        tuple: Lists of power spectra, sine coefficients, cosine coefficients, and mean radii for each frame.
    """
    power_spectra = []
    sine_coeffs_list = []
    cosine_coeffs_list = []
    radii = []

    for coord in coords:
        y_coords = np.asarray(coord[:, 0])
        x_coords = np.asarray(coord[:, 1])
        num_points = len(x_coords)
        if num_points < 2:
            print(f"Skipping frame (too few points: {num_points})")
            continue

        angles = np.zeros(num_points)
        for i in range(num_points):
            radius = np.hypot(x_coords[i], y_coords[i])
            if y_coords[i] > 0:
                angles[i] = np.arccos(x_coords[i] / radius)
            else:

                angles[i] = np.arccos(-x_coords[i] / radius) + np.pi



        sine_coeffs = np.zeros(num_harmonics + 1)
        cosine_coeffs = np.zeros(num_harmonics + 1)
        power_spectrum = np.zeros(num_harmonics + 1)

        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        x_coords = x_coords[sort_idx]
        y_coords = y_coords[sort_idx]
        
        angle_diffs = np.zeros(num_points - 1)
        total_angle_change = 0.0
        for i in range(num_points - 1):
            delta_angle = angles[i + 1] - angles[i]
            if delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            angle_diffs[i] = delta_angle
            total_angle_change += delta_angle

        if total_angle_change < 0:
            angle_diffs = -angle_diffs
            
        mean_radius = 0.0
        for i in range(num_points - 1):
            r1 = np.hypot(x_coords[i], y_coords[i])
            r2 = np.hypot(x_coords[i + 1], y_coords[i + 1])
            mean_radius += 0.5 * angle_diffs[i] * (r1 + r2)
        mean_radius /= (2 * np.pi)
        
        radii.append(mean_radius)
        
    
        
        for harmonic in range(2, num_harmonics + 3):

            sum_sin = 0.0
            sum_cos = 0.0
            for i in range(num_points - 1):
                r1 = np.hypot(x_coords[i], y_coords[i])
                r2 = np.hypot(x_coords[i + 1], y_coords[i + 1])
                theta1 = angles[i]
                theta2 = angles[i + 1]

                sum_sin += 0.5 * angle_diffs[i] * (r1 * np.sin(harmonic * theta1) + r2 * np.sin(harmonic * theta2))
                sum_cos += 0.5 * angle_diffs[i] * (r1 * np.cos(harmonic * theta1) + r2 * np.cos(harmonic * theta2))

            sine_coeffs[harmonic - 2] = sum_sin / (mean_radius * np.pi)
            cosine_coeffs[harmonic - 2] = sum_cos / (mean_radius * np.pi)
            power_spectrum[harmonic - 2] = sine_coeffs[harmonic - 2]**2 + cosine_coeffs[harmonic - 2]**2

        power_spectra.append(power_spectrum)
        sine_coeffs_list.append(sine_coeffs)
        cosine_coeffs_list.append(cosine_coeffs)



    return power_spectra, sine_coeffs_list, cosine_coeffs_list, radii




def lineTension(power_spectra, radii, lt_path, xlsx_path, scale, max_k=10):
    """
    Calculates line tension by fitting the mean square amplitude of boundary fluctuations.

    Args:
        power_spectra (list): List of power spectra arrays.
        radii (list): List of mean radii.
        lt_path (str): Path to save the resulting PDF plot.
        xlsx_path (str): Path to save the extracted fit data.
        max_k (int): Maximum harmonic to include in fit.

    Returns:
        tuple: Lists of x-values, y-values, and the calculated line tension (N).
    """
    k_B = 1.380649e-23    
    temp = 300            

    k = np.arange(2, max_k + 3)  # k vals from 2 to max_k
    x_vals = 1 / (k**2 - 1) 

    plt.figure(figsize=(8,6))

    combined_x = []
    combined_y = []
    xs = []
    ys = []
    rs = []
    
    a2 = []
    
    with PdfPages(lt_path) as pdf:
        
        for i, (spectrum, r) in enumerate(zip(power_spectra, radii)): # loop over all wave coeffs for all particles and frames
            r0 = np.mean(r)
            rs.append(r0 * scale)
            spectrum = np.mean(spectrum, axis=0)

            a2.append(spectrum[0])
   
            
            y_vals = spectrum * r0

            plt.plot(x_vals, y_vals, 'o', label=f"Particle {i+1}")
            

            combined_x.extend(x_vals)
            combined_y.extend(y_vals)
            xs.append(x_vals)
            ys.append(y_vals)

        combined_x = np.array(combined_x)
        combined_y = np.array(combined_y)
        
        
        
        slope, intercept, _, _, _ = linregress(combined_x, combined_y) # fit line to data

        
        kBT_2 = 2 * k_B * temp
        sigma = kBT_2 / (np.pi * slope * scale)  # calculate line tension

        print("Line Tension: ", sigma, " N")



        x_fit = np.linspace(min(x_vals), max(x_vals), 100) # print fit on plot
        y_fit = slope * x_fit + intercept

        
        plt.plot(x_fit, y_fit, 'k--', label=f"Fit: σ = {sigma:.2e} N")
        plt.xlabel(r"$\frac{1}{k^2 - 1}$")
        plt.ylabel(r"$(\langle a_k^2 \rangle + \langle b_k^2 \rangle) \cdot r$ ")
        plt.title(f"Line Tension Fit Across All Particles. Sigma = {sigma:.2e} N")
        plt.grid(True)
        plt.tight_layout()
        # pdf.savefig()
        plt.show()
       
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({
            "x_vals (1 / (k^2 - 1))": combined_x,
            "y_vals (⟨a_k^2 + b_k^2⟩ * r)": combined_y
        }).to_excel(writer, index=False, sheet_name="Fit Data")

        pd.DataFrame({
            "slope": [slope],
            "intercept": [intercept],
            "line tension (N)": [sigma]
        }).to_excel(writer, index=False, sheet_name="Fit Summary")
        
    return xs, ys, rs, sigma, a2
    



def main():
    """
    Main execution function to compute the line tension from tracked domain data.
    """
    
    print("\n\n\nRunning lineTensionCustomFFT.py. Please respond to all GUI prompts...\n\n")
    
    # _ = easygui.buttonbox(
    #     msg="lineTensionCustomFFT.py\n\nThis program calculates the line tension from a multipage .tif file using a boundary fluctuation FFT method from the paper 'Determination of Line Tension in Lipid Monolayers by Fourier Analysis of Capillary Waves' by Benjamin L. Stottrup, Alison M. Heussler, and Tracy A. Bibelnieks (2007)\n\nOutput:\n - xxx_LT.pdf: graph of the FFT wave coefficients from which the line tension is extracted.\n - xxx_LT_DATA.xlsx: Raw data for FFT graph for manual analysis.",
    #     title="Settings",
    #     choices=["Continue"]
    # )
    
    
    xlsx_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_LIGHT_TRACKED.xlsx'
    scale = 0.2222222 * 0.000001
    tif_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_DOMAINS.tif'
    lt_data_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT_DATA.xlsx'
    lt_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT.pdf'

    # xlsx_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_LIGHT_TRACKED_FLTR.xlsx'
    # scale = 0.2222222 * 0.000001
    # tif_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_DOMAINS_FLTR.tif'
    # lt_data_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT_DATA.xlsx'
    # lt_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT.pdf'


    xlsx_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_DARK_TRACKED.xlsx'
    scale = 0.2222222 * 0.000001
    tif_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_DOMAINS_DARK.tif'
    lt_data_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT_DATA.xlsx'
    lt_path = '/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_70MPC_BIN_EFF_LT.pdf'

    
    
    xlsx_path = getXlsx("Select Particle Tracking Data")
    original_tif_path = getTif("Select Binary .tif File")
    
    scale = easygui.multenterbox(
        msg="Enter the number of microns per pixel:",
        title="Settings",
        fields=["Scale (microns/pixel):"],
        values=["0.222222"] 
    )
    
    if scale is None:
        raise Exception("User canceled the input dialog.")

    try:
        scale = float(scale[0]) * 0.000001
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")


    tif_path = isolateDomains(xlsx_path, original_tif_path)
    lt_path = savePdf("Select location to save LT graph data", "_LT", original_tif_path)
    lt_data_path = saveXlsx("Select location to save raw LT data", "_LT_DATA", original_tif_path)




    power_spectra = []
    radii = []
    max_k = 15


    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    for sheet in tqdm(xls.sheet_names, desc="Processing Particles"): # loop over all particle sheets in xlsx file, conduct fft, and append wave coeffs to arrays and calculate LT
        df = xls.parse(sheet)
        
        coords  = calculatePerimeter(tif_path, df)
    
        
        power_spectrum, _, _, r0 = fft(coords, max_k)
        power_spectra.append(power_spectrum)
        radii.append(r0)
        
        
        
    xs, ys, rs, lt, a2 = lineTension(power_spectra, radii, lt_path, lt_data_path, scale, max_k)
    
    
    n = np.arange(3, max_k + 3) 
    
    Nbs, mus, NB_star = dipoleDensity(n, ys, rs, lt, a2)
    
    
    
    Nbs_mean = np.mean(Nbs, axis=0) 
    Nbs_std = np.std(Nbs, axis=0)  
    NBstarmean = np.mean(NB_star)

    

    nmax_vals = np.arange(3, max_k + 3) 
    stars = [NB_star] * len(nmax_vals)
    
    print("\n\n", Nbs_mean, "\n\n", Nbs_std, "\n\n", NB_star, "\n\n")
    
    for i, (Nbm, Nbsd, Nbsm) in enumerate(zip(Nbs_mean, Nbs_std, NB_star)):
        if 0 < Nbm - Nbsd < Nbsm:
         #   print(f"{Nbm} - {Nbsd} < {Nbsm}")
            mu = ((Nbm - Nbsd) * lt / 2)
            elt = lt - mu
            print(f"effective lt for mode {i+1}: {elt},       mu: {mu}")
    
    
    # print(NB_star)
    # print(nmax_vals)

    plt.figure(figsize=(8,5))
    plt.errorbar(nmax_vals, Nbs_mean, yerr=Nbs_std,marker='o', linestyle='-', color='b')
    plt.plot(nmax_vals, stars, marker='o', linestyle='-', color='r')

    plt.xlabel('Maximum Mode Number (nmax)')
    plt.ylabel('Average NB')
    plt.title('Average NB vs. Maximum Mode Number')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
  
    
    # aas = np.array(a2) / np.array(np.mean(ys, axis=0))
    # ns = (nmax_vals**2 -1 )
    
    # slope, intercept, _, _, _ = linregress(ns, aas) # fit line to data
    
    # print(len(a2))
    
    # print(f"\n\na: {aas}\n\nn: {ns}\n\nys: {ys}\n\na: {a2}\n\n\n")

    # plt.figure(figsize=(8,5))
    # plt.plot(ns, aas, marker='o', color='b')
    # x_fit = np.linspace(min(ns), max(ns), 100) # print fit on plot
    # y_fit = slope * x_fit + intercept

    
    # plt.plot(x_fit, y_fit, 'k--')


    # plt.xlabel('Maximum Mode Number (nmax)')
    # plt.ylabel('Average NB')
    # plt.title('Average NB vs. Maximum Mode Number')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    

            
        
    
    
    
    
    

    # effective_lts = np.abs(lt - dds)
    # Nb = 2 * dds / lt
    
    # print(f"eff lt: {effective_lts}, Nb: {Nb}\n")


























if __name__ == "__main__":
    main()