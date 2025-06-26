import os
from skimage import io
import pandas as pd
import numpy as np
import isolateSingleDomain
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import linregress
import math


def sigfigs(x, sigfigs):
    if x == 0:
        return 0
    return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)



def calculateAngle(cx, cy, row, col):
    """
    Returns angle in radians between the vector from center (cy, cx)
    to point (row, col) and the positive x-axis.
    """
    dy = row - cy  # row increases downward
    dx = col - cx  # col increases rightward

    angle = np.arctan2(dy, dx) % (2 * np.pi)
    return angle
    


def calculatePerimeter(tif_path, data_path):
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)
    tif_stack = io.imread(tif_path).astype(np.uint8)
    
    delta_r = []
    all_angles = []
    mean_r = []
        
    for i, row in tqdm(df_all.iterrows(), total=df_all.shape[0], desc="Processing Particles"):
        domain_color = int(row["Domain_Color"])
        area = int(row["Area"])
        centroid_r = int(row["Centroid_Y"])
        centroid_c = int(row["Centroid_X"])
        

        r = np.sqrt(area / np.pi) 
        mean_r.append(r)

        
        try:
            frame_slice = tif_stack[i]
        except:
            continue
        
        # frame_slice = tif_stack

        domain_pixels = np.column_stack(np.where(frame_slice == domain_color))


        radii = []
        angles = []
        for row_pixel, col_pixel in domain_pixels:
            dy = row_pixel - centroid_r
            dx = col_pixel - centroid_c
            distance = np.sqrt(dy ** 2 + dx ** 2)
            dr = r - distance

            angle = calculateAngle(centroid_r, centroid_c, row_pixel, col_pixel)
            
            radii.append((dr * 1e-6 * 0.2222))
            angles.append(angle)
        

           
        radii = np.array(radii)
        angles = np.array(angles)
        sort_idx = np.argsort(angles)
        theta_sorted = angles[sort_idx]
        radii_sorted = radii[sort_idx]
        px = domain_pixels[sort_idx]

        delta_r.append(radii_sorted)
        all_angles.append(theta_sorted)

        print("frame_slice: ", frame_slice)
        for i, (rad, ang, (row, col)) in enumerate(zip(radii_sorted, theta_sorted, px)):
            if i % 10 == 0: print(f"{i+1}: ({row}, {col}): {sigfigs(np.abs(rad),3)} at θ = {ang}       ({row - centroid_r},{col - centroid_c})     centroid: ({centroid_r, centroid_c})")
            frame_slice[row][col] = i + 101
            frame_slice[centroid_r][centroid_c] = 255
        # print("frame_slice: ", frame_slice)
            
        # print(f"\nR - RADII: {r - radii}\n" )
            
    
    return delta_r, all_angles, mean_r



def fft(delta_r, angles, fft_path, max_k=10):
    thetas = []
    radiii = []
    fft = []
    

    num_images = len(angles[0])
    
    a_ks = np.zeros((num_images, max_k-1))
    b_ks = np.zeros((num_images, max_k-1))

    for i in range(len(delta_r)):
        radii = delta_r[i]   
        theta = angles[i]  
        
 
        theta_uniform = np.linspace(0, 2*np.pi, num_images, endpoint=False)
        radii_uniform = np.interp(theta_uniform, theta, radii)


        fft_coeffs = np.fft.fft(radii_uniform)
        radii_reconstructed = np.fft.ifft(fft_coeffs).real
        
        thetas.append(theta_uniform)
        radiii.append(radii_uniform)
        fft.append(radii_reconstructed)
        
        N = len(radii_uniform)
        
        for k in range(1, max_k):
            coeff = fft_coeffs[k]
            a_k = 2 * np.real(coeff) / N
            b_k = -2 * np.imag(coeff) / N

            a_ks[i, k - 1] = a_k 
            b_ks[i, k - 1] = b_k 

        
        

        
        
        
        
    with PdfPages(fft_path) as pdf:
        for i, (th, r, rfft) in enumerate(tqdm(zip(thetas, radiii, fft), total=len(delta_r), desc="Saving Plots")):
            plt.figure(figsize=(8, 4))
            plt.plot(th, r, label='Original Δr(θ)', color='blue')
            plt.plot(th, rfft, '--', label='Reconstructed (from FFT)', color='red')
            plt.xlabel('Angle θ (radians)')
            plt.ylabel('Δr (pixels)')
            plt.legend()
            plt.title('Δr(θ): Original vs. Reconstructed from FFT')
            plt.tight_layout()
            plt.grid(True)
            pdf.savefig()
            plt.close()

            k_vals = np.arange(num_images)
            plt.figure(figsize=(6, 4))
            plt.stem(k_vals[2:], np.abs(fft_coeffs[2:]), basefmt=" ")
            plt.xlabel("k (wave number)")
            plt.ylabel("|FFT coefficient|")
            plt.title(f"Frame {i}: FFT Magnitude Spectrum")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            
    mean_ak = np.mean(a_ks, axis=0)  
    mean_bk = np.mean(b_ks, axis=0) 


    for k in range(1, max_k):
        print(f"k={k+1}: ⟨aₖ⟩ = {mean_ak[k-1]:.5f}, ⟨bₖ⟩ = {mean_bk[k-1]:.5f}")
    
    

    return fft_coeffs, a_ks, b_ks




def lineTension(a_ks, b_ks, mean_r, max_k=10):
    k_B = 1.380649e-23    
    T = 298              
    r = np.mean(mean_r) 

    mean_ak_sq = np.mean(a_ks**2, axis=0)  
    mean_bk_sq = np.mean(b_ks**2, axis=0) 
    
    k = np.arange(2, max_k + 1)


    x_vals = 1 / (k**2 - 1)
    y_vals = (mean_ak_sq + mean_bk_sq) * r 


    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)

    sigma = (2 * k_B * T) / (np.pi * slope)

    plt.close()
    plt.plot(x_vals, y_vals, 'o', label="Data")
    plt.plot(x_vals, slope * x_vals + intercept, 'r--', label=f"Fit: σ = {sigma:.2e} N")
    plt.xlabel(r"$\frac{1}{k^2} - 1$")
    plt.ylabel(r"$(\langle a_k^2 \rangle + \langle b_k^2 \rangle) \cdot r$")
    plt.title("Line Tension Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # print(x_vals, slope * x_vals + intercept, r)



def main():
    tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_DOMAIN17.tif"
    data_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/6.5mNm_DOMAIN17.xlsx"
    
    # tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/wave_circle1_DOMAIN1.tif"
    # data_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/wave_circle1_DOMAIN1.xlsx"
    
    data_path, tif_path = isolateSingleDomain.getFiles()
    tif_path, data_path = isolateSingleDomain.isolateDomain(data_path, tif_path)
    
    # tif_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/fft_DOMAIN83.tif"
    # data_path = "/Users/myleskoppelman/Documents/University Of Minnesota/Augsburg Internship 2025/fft_DOMAIN83.xlsx"
    
    delta_r, angles, mean_r = calculatePerimeter(tif_path, data_path)
    
    path, filename = os.path.split(tif_path)
    name, _ = os.path.splitext(filename)
    


    pdf_path = os.path.join(path, f"{name}_DELTA_R.pdf")
    fft_path = os.path.join(path, f"{name}_FFT.pdf")
    
    # print(delta_r[0], angles[0])
    
    with PdfPages(pdf_path) as pdf:
        for i, (dr, angle) in enumerate(tqdm(zip(delta_r, angles), total=len(delta_r), desc="Saving Plots")):
            plt.figure(figsize=(6, 4))
            plt.scatter(angle, dr, s=5, alpha=0.6)
            plt.xlabel('Angle θ (radians)')
            plt.ylabel('Δr = radius - distance')
            plt.title(f'dr vs θ for frame/particle {i}')
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
    # print(len(delta_r[0]), len(angles[0]))
    coeffs, a_ks, b_ks = fft(delta_r, angles, fft_path, 10)
    lineTension(a_ks, b_ks, mean_r, 10)
    


        
if __name__ == "__main__":
    main()
    