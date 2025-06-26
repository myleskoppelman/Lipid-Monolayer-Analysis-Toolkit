import tifffile
from skimage import io
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, disk, binary_erosion
from scipy.ndimage import binary_fill_holes
import easygui
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
    
    
    
    

def main():
    data_path = easygui.fileopenbox(
        msg="Select Particle Tracking Data",
        default="Data#",
        filetypes=["*.xlsx"]
    )
    if not data_path:
        raise Exception("No file selected.")

    
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)
    
    rads = []
    speeds = []

    for _, row in df_all.iterrows():
        
        try:
            rad= int(row["Head Radius (meters)'"])
            rads.append(rad)
        except:
            continue
        
        try:
            speed = int(row(["dL/dt (m/s)"]))
            speeds.append(speed)
        except:
            continue
        
    print(rads, speeds)
    plt.plot(speeds, rads, 'k--')
    plt.xlabel("rads(m)")
    plt.ylabel("m/s")
    plt.title(f"Radius v.s. Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
