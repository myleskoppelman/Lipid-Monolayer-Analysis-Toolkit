import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *
def calculate_kinematics(df, scale=1.0, fps=1.0):
    """
    Given a dataframe for one domain, compute velocity, acceleration, and jerk using
    central differences. Assumes frames are in order.

    df must have 'Centroid_X' and 'Centroid_Y' columns.
    scale: conversion factor from pixels to physical units (e.g., microns per pixel)
    fps: frames per second
    """
    dt = 1.0 / fps

    # Apply scale
    x = df["Centroid_X"].to_numpy()
    y = df["Centroid_Y"].to_numpy()

    N = len(x)

    # Initialize arrays
    vx = np.zeros(N)
    vy = np.zeros(N)
    ax = np.zeros(N)
    ay = np.zeros(N)
    jx = np.zeros(N)
    jy = np.zeros(N)

    # --- Velocity (central difference) ---
    for i in range(1, N-1):
        vx[i] = (x[i+1] - x[i-1]) / (2*dt)
        vy[i] = (y[i+1] - y[i-1]) / (2*dt)
    # forward/backward for endpoints
    vx[0] = (x[1] - x[0]) / dt
    vy[0] = (y[1] - y[0]) / dt
    vx[-1] = (x[-1] - x[-2]) / dt
    vy[-1] = (y[-1] - y[-2]) / dt

    # --- Acceleration (central difference of velocity) ---
    for i in range(1, N-1):
        ax[i] = (vx[i+1] - vx[i-1]) / (2*dt)
        ay[i] = (vy[i+1] - vy[i-1]) / (2*dt)
    ax[0] = (vx[1] - vx[0]) / dt
    ay[0] = (vy[1] - vy[0]) / dt
    ax[-1] = (vx[-1] - vx[-2]) / dt
    ay[-1] = (vy[-1] - vy[-2]) / dt

    # --- Jerk (central difference of acceleration) ---
    for i in range(1, N-1):
        jx[i] = (ax[i+1] - ax[i-1]) / (2*dt)
        jy[i] = (ay[i+1] - ay[i-1]) / (2*dt)
    jx[0] = (ax[1] - ax[0]) / dt
    jy[0] = (ay[1] - ay[0]) / dt
    jx[-1] = (ax[-1] - ax[-2]) / dt
    jy[-1] = (ay[-1] - ay[-2]) / dt

    # Magnitudes
    v = np.sqrt(vx**2 + vy**2)
    a = np.sqrt(ax**2 + ay**2)
    j = np.sqrt(jx**2 + jy**2)
    
    v_angle = np.degrees(np.arctan2(vy, vx))
    a_angle = np.degrees(np.arctan2(ay, ax))
    j_angle = np.degrees(np.arctan2(jy, jx))

    # Take absolute angles (0–180°)
    v_angle = np.abs(v_angle)
    a_angle = np.abs(a_angle)
    j_angle = np.abs(j_angle)
    

    # Save to dataframe
    df["Vel_X"] = vx
    df["Vel_Y"] = vy
    df["Velocity"] = v
    df["Vel_Angle"] = v_angle
    df["Acc_X"] = ax
    df["Acc_Y"] = ay
    df["Acceleration"] = a
    df["Acc_Angle"] = a_angle
    df["Jerk_X"] = jx
    df["Jerk_Y"] = jy
    df["Jerk"] = j
    df["Jerk_Angle"] = j_angle

    return df



def process_xlsx(input_file, output_file, scale, fps):
    xls = pd.ExcelFile(input_file)

    processed_sheets = {}

    # tqdm progress bar over sheet names
    for sheet_name in tqdm(xls.sheet_names, desc="Processing domains", unit="sheet"):
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = calculate_kinematics(df, scale=scale, fps=fps)
        processed_sheets[sheet_name] = df

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n Saved kinematic analysis to {output_file}")








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
    
    
    xlsx_input = getXlsx("Select Particle Tracking Data")
    xlsx_output = saveXlsx("Select Location to Save Particle Data", "_KINEMATICS", xlsx_input)
    
    process_xlsx(xlsx_input, xlsx_output, scale, fps)
    
    


if __name__ == "__main__":
    main()


