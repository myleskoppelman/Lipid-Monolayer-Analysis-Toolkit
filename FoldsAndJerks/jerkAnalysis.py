import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import binned_statistic
from utils import *

def plotData(filename, fps, scale, 
             output_velocity_pdf, output_jerk_pdf,
             output_angle_pdf, output_angle_vs_mag_pdf):
    """
    Create four multipage PDFs:
      1. Velocity vs Time (sizemographs)
      2. Jerk magnitude histograms
      3. Jerk angle distributions
      4. Jerk angle vs Jerk magnitude correlation (binned)
    """
    xls = pd.ExcelFile(filename)
    
    all_jerks = []
    all_angles = []
    all_angle_mag_pairs = []

    with PdfPages(output_velocity_pdf) as vel_pdf, \
         PdfPages(output_jerk_pdf) as jerk_pdf, \
         PdfPages(output_angle_pdf) as angle_pdf, \
         PdfPages(output_angle_vs_mag_pdf) as anglemag_pdf:

        # loop with progress bar
        for sheet_name in tqdm(xls.sheet_names, desc="Processing sheets", unit="sheet"):
            df = pd.read_excel(filename, sheet_name=sheet_name)

            # --------- Velocity vs Time ---------
            time = df["Frame"] / fps
            velocity = df["Velocity"]
            vx = df["Vel_X"]
            vy = df["Vel_Y"]

            plt.figure(figsize=(10, 4))
            plt.plot(time, velocity, lw=0.8, color="black")
            plt.plot(time, vx, lw=0.8, color="red")
            plt.plot(time, vy, lw=0.8, color="blue")
            plt.xlabel("time (s)")
            plt.ylabel("velocity (µm/s)")
            plt.title(f"Velocity vs Time - {sheet_name}")
            plt.legend(["Velocity Magnitude", "X Velocity", "Y Velocity"])
            plt.tight_layout()
            vel_pdf.savefig()
            plt.close()

            # --------- Jerk Magnitude Histogram ---------
            jerk_mag = df["Jerk"].to_numpy()
            jerk_mag = jerk_mag[~np.isnan(jerk_mag)]
            all_jerks.append(jerk_mag)

            plt.figure(figsize=(6, 4))
            plt.hist(jerk_mag, bins=int(np.sqrt(len(jerk_mag))), density=True,
                     color="white", edgecolor="black")
            plt.xlabel("jerk magnitude (µm/s³)")
            plt.ylabel("frequency")
            plt.title(f"Jerk Magnitude Distribution - {sheet_name}")
            plt.tight_layout()
            jerk_pdf.savefig()
            plt.close()

            # --------- Jerk Angle Histogram ---------
            jerk_ang = np.abs(df["Jerk_Angle"].to_numpy())
            jerk_ang = jerk_ang[~np.isnan(jerk_ang)]
            all_angles.append(jerk_ang)

            plt.figure(figsize=(6, 4))
            plt.hist(jerk_ang, bins=np.arange(0,71,2), density=True,
                     color="white", edgecolor="black")
            plt.xlabel("angle |θ| (°)")
            plt.ylabel("frequency pθ (1/°)")
            plt.title(f"Jerk Angle Distribution - {sheet_name}")
            plt.tight_layout()
            angle_pdf.savefig()
            plt.close()

            # --------- Jerk Angle vs Magnitude (Binned) ---------
 
            mask = ~df["Jerk"].isna() & ~df["Jerk_Angle"].isna()
            jerk_mag_nonan = df.loc[mask, "Jerk"].to_numpy()
            jerk_ang_nonan = np.abs(df.loc[mask, "Jerk_Angle"].to_numpy())
            all_angle_mag_pairs.append(np.column_stack([jerk_mag_nonan, jerk_ang_nonan]))
            bins = np.logspace(np.log10(max(1e-6, jerk_mag_nonan.min())),np.log10(jerk_mag_nonan.max()), 20)
            bin_means, bin_edges, _ = binned_statistic(jerk_mag_nonan, jerk_ang_nonan, statistic='mean', bins=bins)
            bin_counts, _ = np.histogram(jerk_mag_nonan, bins=bins)
            bin_stds, _, _ = binned_statistic(jerk_mag_nonan, jerk_ang_nonan, statistic='std', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_err = bin_stds / np.sqrt(bin_counts)

            plt.figure(figsize=(6, 4))
            plt.errorbar(bin_centers, bin_means, yerr=bin_err, fmt='s', color="black", capsize=3)
            plt.xscale("log")
            plt.xlabel("magnitude l (µm/s³)")
            plt.ylabel("angle |θ| (°)")
            plt.title(f"Jerk Angle vs Magnitude - {sheet_name}")
            plt.tight_layout()
            anglemag_pdf.savefig()
            plt.close()

        # --------- Pooled Distributions ---------
        all_jerks = np.concatenate(all_jerks)
        all_angles = np.concatenate(all_angles)
        all_angle_mag_pairs = np.vstack(all_angle_mag_pairs)

        # Jerk magnitude cumulative
        plt.figure(figsize=(8, 5))
        plt.hist(all_jerks, bins=int(np.sqrt(len(all_jerks))), density=True, cumulative=True, histtype="step", color="black")
        plt.xlabel("jerk magnitude (µm/s³)")
        plt.ylabel("cumulative probability")
        plt.title("Cumulative Jerk Magnitude Distribution (All Sheets)")
        plt.tight_layout()
        jerk_pdf.savefig()
        plt.close()

        # Jerk angle pooled
        plt.figure(figsize=(8, 5))
        plt.hist(all_angles, bins=np.arange(0,71,2), density=True,
                 color="white", edgecolor="black")
        plt.xlabel("angle |θ| (°)")
        plt.ylabel("frequency pθ (1/°)")
        plt.title("Cumulative Jerk Angle Distribution (All Sheets)")
        plt.tight_layout()
        angle_pdf.savefig()
        plt.close()

        # Jerk angle vs magnitude pooled (with binning)
        mags = all_angle_mag_pairs[:,0]
        angs = all_angle_mag_pairs[:,1]
        bins = np.linspace(mags.min(), mags.max(), 25)
        bin_means, bin_edges, _ = binned_statistic(mags, angs, statistic='mean', bins=bins)
        bin_counts, _ = np.histogram(mags, bins=bins)
        bin_stds, _, _ = binned_statistic(mags, angs, statistic='std', bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_err = bin_stds / np.sqrt(bin_counts)

        plt.figure(figsize=(8, 5))
        plt.errorbar(bin_centers, bin_means, yerr=bin_err, fmt='s', color="black", capsize=3)

        plt.xlabel("magnitude l (µm/s³)")
        plt.ylabel("angle |θ| (°)")
        plt.title("Jerk Angle vs Magnitude (All Sheets)")
        plt.tight_layout()
        anglemag_pdf.savefig()
        plt.close()

    print(f"\nSaved sizemographs to {output_velocity_pdf}")
    print(f"Saved jerk histograms + cumulative to {output_jerk_pdf}")
    print(f"Saved jerk angle distributions to {output_angle_pdf}")
    print(f"Saved jerk angle vs magnitude plots to {output_angle_vs_mag_pdf}")








def main():
    
    settings = easygui.multenterbox(
        msg="Enter scale and FPS",
        title="Settings",
        fields=["Microns per Pixel", "Frames per Second"],
        values= ["0.222222", "20"]
    )
    try:
        px = float(settings[0])
        fps = float(settings[1])
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")

    scale = 1e-6 * px
    
    
    xlsx_input = getXlsx("Select Particle Tracking Data with Kinematics")
    velocity_pdf = savePdf("Select Location to Save Particle Velocity Data", "_VEL", xlsx_input)
    jerk_pdf = savePdf("Select Location to Save Particle Jerk Data", "_JERK", xlsx_input)
    jerk_angle_pdf = savePdf("Select Location to Save Particle Jerk Data", "_JERK_ANGLE", xlsx_input)
    jerk_mag_pdf = savePdf("Select Location to Save Particle Jerk Data", "_ANGLE_MAGNITUDE", xlsx_input)
    
    plotData(xlsx_input, fps, scale, velocity_pdf, jerk_pdf, jerk_angle_pdf, jerk_mag_pdf)
    
     
    
if __name__ == "__main__":
    main()
    