import os
from skimage import io
from dipoleDensity import *
import pandas as pd
import numpy as np
import isolateAllDomains
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import linregress

scale = 1e-6 * 0.222222

def calculateAngle(cx, cy, row, col):
    """
    Returns angle in radians between the vector from center (cy, cx)
    to point (row, col) and the positive x-axis.
    """
    dy = row - cy 
    dx = col - cx  

    angle = np.arctan2(dy, dx) % (2 * np.pi)
    return angle




def calculatePerimeter(tif_path, data_path):
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    tif_stack = io.imread(tif_path).astype(np.uint8)
    
    delta_rs = []
    all_angles = []
    mean_rs = []
        
    for sheet in tqdm(xls.sheet_names, desc="Processing Sheets"):
        df_sheet = xls.parse(sheet) 
        delta_r = []
        angles = []
        mean_r = []
        for i, row in df_sheet.iterrows():
            frame = int(row["Frame"]) - 1
            if not (0 <= frame < len(tif_stack)):
                # print(f"Skiping invalid frame index: {frame}")
                continue
            domain_color = int(row["Domain_Color"])
            area = int(row["Area"])
            centroid_r = int(row["Centroid_Y"])
            centroid_c = int(row["Centroid_X"])
            bbx = int(row["BBox_X"]) 
            bby = int(row["BBox_Y"]) 
            bbw = int(row["BBox_W"]) 
            bbh = int(row["BBox_H"])
            

            r = np.sqrt(area / np.pi)
            mean_r.append(r) # * scale
            
            padding = 5
            img_height, img_width = tif_stack[frame].shape

            x_start = max(bbx - padding, 0)
            y_start = max(bby - padding, 0)
            x_end   = min(bbx + bbw + padding, img_width)
            y_end   = min(bby + bbh + padding, img_height)

            try:
                frame_slice = tif_stack[frame][y_start:y_end, x_start:x_end]
            except:
                continue
            yy, xx = np.where(frame_slice == domain_color)

            domain_pixels = np.column_stack((yy + y_start, xx + x_start))


            frame_delta_r = []
            angle = []
            for row_pixel, col_pixel in domain_pixels:
                dy = row_pixel - centroid_r
                dx = col_pixel - centroid_c
                distance = np.sqrt(dy ** 2 + dx ** 2)
                dr = r - distance
                
                frame_angle = calculateAngle(centroid_r, centroid_c, row_pixel, col_pixel)

                frame_delta_r.append(dr * scale)
                angle.append(frame_angle)

            
            frame_delta_r = np.array(frame_delta_r)
            angle = np.array(angle)
            sort_idx = np.argsort(angle)
            
            frame_angle_sorted = angle[sort_idx] 
            frame_delta_r_sorted = frame_delta_r[sort_idx] 

            delta_r.append(frame_delta_r_sorted)
            angles.append(frame_angle_sorted)
            
        delta_rs.append(delta_r)
        all_angles.append(angles)
        mean_rs.append(mean_r)
                
        
    return delta_rs, all_angles, mean_rs 



def fft(delta_r, angles, fft_path, max_k=10):
    frame_angles = []
    frame_delta_ri = []
    fft = []
    
    all_aks = []
    all_bks = []
    
    for i_particle in range(len(delta_r)):
        particle_delta_r = delta_r[i_particle]
        particle_angles = angles[i_particle]
        
        num_images = len(particle_delta_r)
    
        a_ks = np.zeros((num_images, max_k-1))
        b_ks = np.zeros((num_images, max_k-1))


        for i_frame in range(len(particle_delta_r)):
            frame_delta_r = np.array(particle_delta_r[i_frame])
            frame_angle = np.array(particle_angles[i_frame])
            if frame_delta_r.size == 0 or frame_angle.size == 0:
                # print(f"Skipping empty frame {i_frame} of particle {i_particle}")
                continue
            
            N = max_k 
    
            frame_angle_uniform = np.linspace(0, 2*np.pi, N, endpoint=False)
            frame_delta_r_uniform = np.interp(frame_angle_uniform, frame_angle, frame_delta_r)

            fft_coeffs = np.fft.fft(frame_delta_r_uniform)
            frame_delta_r_reconstructed = np.fft.ifft(fft_coeffs).real
            
            frame_angles.append(frame_angle_uniform)
            frame_delta_ri.append(frame_delta_r_uniform)
            fft.append(frame_delta_r_reconstructed)
            
            
            for k in range(1, max_k):
                coeff = fft_coeffs[k]
                a_k = 2 * np.real(coeff) / N
                b_k = -2 * np.imag(coeff) / N

                a_ks[i_frame, k - 1] = a_k 
                b_ks[i_frame, k - 1] = b_k 
 
        all_aks.append(a_ks)
        all_bks.append(b_ks)
    
    return fft_coeffs, all_aks, all_bks




def lineTension(all_aks, all_bks, mean_r, lt_path, xlsx_path, max_k=10):
    k_B = 1.380649e-23    
    temp = 298              
    
    k = np.arange(2, max_k + 1)  

    x_vals = 1 / (k**2 - 1)

    plt.figure(figsize=(8,6))
    

    combined_x = []
    combined_y = []
    xs = []
    ys = []
    mean_rs = []
    with PdfPages(lt_path) as pdf:
        for i_particle, (a_ks, b_ks, rs) in enumerate(zip(all_aks, all_bks, mean_r)):
            dr = np.mean(rs)
            mean_rs.append(dr)

            mean_ak_sq = np.mean(a_ks**2, axis=0)
            mean_bk_sq = np.mean(b_ks**2, axis=0)

            y_vals = (mean_ak_sq + mean_bk_sq) * dr 
            if np.isnan(y_vals).any():
                # print(f"Skipping particle {i_particle+1} due to NaNs in y_vals")
                continue

            plt.plot(x_vals, y_vals, 'o', label=f"Particle {i_particle+1}")

            combined_x.extend(x_vals)
            combined_y.extend(y_vals)
            xs.append(x_vals)
            ys.append(y_vals)

        combined_x = np.array(combined_x)
        combined_y = np.array(combined_y)
        dr = np.mean(mean_rs)
        print(dr)
        slope, intercept, _, _, _ = linregress(combined_x, combined_y)
        sigma = (2 * k_B * temp) / (np.pi * slope * dr)
        

        print("Line Tension: ", sigma, " N")


        x_fit = np.linspace(min(x_vals), max(x_vals), 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, 'k--', label=f"Fit: σ = {sigma:.2e} N")
        plt.xlabel(r"$\frac{1}{k^2 - 1}$")
        plt.ylabel(r"$(\langle a_k^2 \rangle + \langle b_k^2 \rangle) \cdot r$ ")
        plt.title(f"Line Tension Fit Across All Particles. Sigma = {sigma:.2e} N")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.show()
       
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({
            "x_vals (1 / (k^2 - 1))": combined_x,
            "y_vals (⟨a_k^2 + b_k^2⟩ * r)": combined_y
        }).to_excel(writer, index=False, sheet_name="Fit Data")

        pd.DataFrame({
            "dr (mean radius)": [dr],
            "slope": [slope],
            "intercept": [intercept],
            "line tension (N)": [sigma]
        }).to_excel(writer, index=False, sheet_name="Fit Summary")
        
    return xs, ys, mean_rs, sigma

    




def main():
    tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_DOMAINS.tif"
    data_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_LIGHT_TRACKED.xlsx"
    
    # data_path, tif_path = isolateAllDomains.getFiles()
    # tif_path = isolateAllDomains.isolateDomains(data_path, tif_path)
    
    # tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/fft_DOMAIN83.tif"
    # data_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/fft_DOMAIN83.xlsx"
    
    delta_r, angles, mean_r = calculatePerimeter(tif_path, data_path)
    
    path, filename = os.path.split(tif_path)
    name, _ = os.path.splitext(filename)
    


    lt_path = os.path.join(path, f"{name}_LT.pdf")
    fft_path = os.path.join(path, f"{name}_FFT.pdf")
    xlsx_path = os.path.join(path, f"{name}_LT_DATA.xlsx")
    
    
    
    # with PdfPages(pdf_path) as pdf:
    #     for i, (dr, angle) in enumerate(tqdm(zip(delta_r, angles), total=len(delta_r), desc="Saving Plots")):
    #         plt.figure(figsize=(6, 4))
    #         plt.scatter(angle, dr, s=5, alpha=0.6)
    #         plt.xlabel('Angle θ (radians)')
    #         plt.ylabel('Δr = radius - distance')
    #         plt.title(f'dr vs θ for frame/particle {i}')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()
            
            
    _, a_ks, b_ks = fft(delta_r, angles, fft_path, 15)
    xs, ys, rs, lt = lineTension(a_ks, b_ks, mean_r, lt_path, xlsx_path, 15)
    
    
    
    dds = dipoleDensity(xs, ys, rs, lt)
    
    print(dds)
    effective_lts = np.abs(lt - dds)
    
    x = np.arange(0, len(effective_lts))
    lts = [lt] * len(effective_lts)
    
    
    plt.plot(x, effective_lts, 'k--', label=f"FT vs EFFECTIVE LT")
    plt.plot(x, lts, 'o--')
    plt.xlabel("Effective Line Tension (N)")
    plt.ylabel("Line Tension (N)")
    plt.title(f"FT vs EFFECTIVE LT")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    print(effective_lts)


        
if __name__ == "__main__":
    main()
    