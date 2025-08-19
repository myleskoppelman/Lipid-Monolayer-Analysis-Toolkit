import os
import easygui
import numpy as np
import matplotlib.pyplot as plt
from arcLength import *
from drawEllipses import *
from fitEllipse import *


''' [lineTension.py] Last Updated: 6/6/2025 by Myles Koppelman '''

def getFiles() -> tuple[str, str]:
    """
    
    Opens file dialogs for the user to select the input data file, binary .tif file, 
    and the output .tif file path for saving.

    Returns
    -------
    tuple of str
        A tuple containing:
        - data_path: Path to the selected Excel data file (.xlsx).
        - tif_path: Path to the selected binary .tif file.
        - save_path: Path to save the output .tif file.

    Raises
    ------
    Exception
        If the user does not select a file in any of the dialogs.
    """
    data_path = easygui.fileopenbox(
        msg="Select Particle Tracking Data",
        default="Data#",
        filetypes=["*.xlsx"]
    )
    if not data_path:
        raise Exception("No file selected.")

    
    tif_path = easygui.fileopenbox(
        msg="Select Binary .tif File",
        default="Data#",
        filetypes=["*.tif"]
    )
    if not tif_path:
        raise Exception("No file selected.")
    
    return data_path, tif_path



def getSaveFiles(tif_path) -> tuple[str, str]:
    """
    
    Opens file dialogs for the user to select the input data file, binary .tif file, 
    and the output .tif file path for saving.

    Returns
    -------
    tuple of str
        A tuple containing:
        - save_path: Path to save the output .tif file.

    Raises
    ------
    Exception
        If the user does not select a file in any of the dialogs.
    """
    
    path, filename = os.path.split(tif_path)
    nm, _ = os.path.splitext(filename)
    

    save_path = easygui.filesavebox(
        msg="Save Output .tif File",
        default=os.path.join(path, f"{nm}_LT.tif"),
        filetypes=["*.tif"]
    )
    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.tif'):
        save_path += '.tif'
        
        
    data_save_path = easygui.filesavebox(
        msg="Save Output .xlsx File",
        default=os.path.join(path, f"{nm}_LT_DATA.xlsx"),
        filetypes=["*.xlsx"]
    )
    if not data_save_path:
        raise Exception("No output file selected.")
    
    if data_save_path is not None and not data_save_path.lower().endswith('.xlsx'):
        data_save_path += '.xlsx'
    
    return save_path, data_save_path






def lineTension(data_path, tif_path, save_path, data_save_path):
    """
    Calculates the line tension of 'bola' structures formed between monolayers of certain liquids
    based on the methodology in Bischof & Wilke (2017).

    The function:
    1. Prompts the user for the head index, pixel size (microns per pixel), and subsurface viscosity.
    2. Reads time-dependent data of the structure from the given `.tif` file and ellipse data.
    3. Calculates:
        - Arc lengths
        - Major and minor ellipse axes
        - Time-dependent head radii
        - Line tension as a function of time
    4. Computes weighted average and median line tension values.
    5. Generates plots for:
        - Arc length over time
        - Head radius over time
        - Line tension over time
    6. Saves output images with ellipses fitted to the data.
    7. Saves all time-series data into a single `.xlsx` file.

    Parameters:
    -----------
    data_path : str
        Path to the data file (e.g., .xlsx) containing ellipse fitting data.

    tif_path : str
        Path to the `.tif` image file of the monolayer structure over time.

    save_path : str
        Directory path where output images with ellipse overlays and the Excel file will be saved.

    Returns:
    --------
    None
        Prints the weighted average and median line tension values, shows plots, and saves data.

    Notes:
    ------
    - The function requires `arcLength`, `fitEllipse`, and `drawEllipses` helper functions.
    - Uses scikit-image, matplotlib, numpy, pandas, and easygui for data handling and visualization.

    Reference:
    ----------
    Bischof, A. A., & Wilke, N. (2017). Molecular determinants for the line tension of coexisting liquid phases in monolayers.
    """

    settings = easygui.multenterbox(
        msg="Enter index of the 'head' of the bola, pixel size, frames per seconds, and subsurface viscosity.",
        title="Settings",
        fields=["Head Index: ", "Microns per Pixel", "Frames per Second", "Subsurface Viscosity"],
        values= ["1", "0.222222", "20", "0.001"]
    )
    try:
        head = int(settings[0])
        scale = float(settings[1])
        fps = float(settings[2])
        viscosity = float(settings[3]) # Pascal Seconds or Newtons per Meter^2
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")

    pixel_size = 1e-6 * scale

    # Get data
    arc_lengths, poly_funcs = arcLength(data_path, tif_path, head) # get arclength of bola
    _, majors, minors = fitEllipse(data_path, tif_path, head) # Get major/minor axis of head of bola in 

    arc_lengths = np.array(arc_lengths) * pixel_size # scale data
    majors = np.array(majors) * pixel_size 
    minors = np.array(minors) * pixel_size

    x1 = np.arange(len(arc_lengths)) / fps # scale time axis 
    x2 = np.arange(len(majors)) / fps


    a = np.sqrt((majors/2)*(minors/2)) # calculate radius of head

    coeffs = np.polyfit(x1, arc_lengths, deg=1) # conduct fit on results from arclength
    poly = np.poly1d(coeffs)
    y_fit = poly(x1)

    dL_dt = poly.deriv() # take derivative of the fit to find change in arc length over time
    f = 8
    line_tension = f * viscosity * dL_dt * a # calculate line tension 
    line_tension = np.array(line_tension)

    center = np.median(line_tension) # calculate median line tension
    deviations = line_tension - center
    sigma = 1.0
    weights = np.exp(- (deviations)**2 / (2 * sigma**2))
    lt2 = np.average(line_tension, weights=weights) # conduct weighted average on line tension

    avg_lt = [lt2] * len(x2)
    med_lt = [center] * len(x2)

    drawEllipses(data_path, tif_path, save_path, poly_funcs) # draw arc length and radial ellipses

    print(f"\n\nWeighted Average Line Tension: {lt2} Newtons\nMedian Line Tension: {center} Newtons\n\n")
   

    # Plot 1: Arc lengths
    plt.figure(figsize=(8,4))
    plt.plot(x1, arc_lengths, 'r', linestyle='-', label='Arc Lengths')
    plt.plot(x1, y_fit, 'b', linestyle='-', label='Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Arc length (microns)')
    plt.title('Arc Length over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Head radius
    plt.figure(figsize=(8,4))
    plt.plot(x2, a, 'g', linestyle='-', label='Head Radius')
    plt.xlabel('Time (s)')
    plt.ylabel('Radius (microns)')
    plt.title('Head Radius over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 3: Line tension
    plt.figure(figsize=(8,4))
    plt.plot(x2, line_tension, 'r', marker='o', linestyle='-', label='Line Tension')
    plt.plot(x2, avg_lt, 'b', linestyle='-', label='Weighted Average Line Tension')
    plt.plot(x2, med_lt, 'g', linestyle='-', label='Median Line Tension')
    plt.xlabel('Time (s)')
    plt.ylabel('Line Tension (N)')
    plt.title('Line Tension over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    with pd.ExcelWriter(data_save_path) as writer: # write plot data to .xlsx file
        df1 = pd.DataFrame({
            'Time (s)': x1,
            'Arc Length (meters)': arc_lengths,
            'Arc Length Fit (meters)': y_fit,
            'dL/dt (m/s)': [dL_dt] * len(x1)
        })
        df1.to_excel(writer, sheet_name='Arc Lengths', index=False)

        df2 = pd.DataFrame({
            'Time (s)': x2,
            'Head Radius (meters)': a,
            'Line Tension (N)': line_tension,
            'Weighted Avg Line Tension (N)': avg_lt,
            'Median Line Tension (N)': med_lt
        })
        df2.to_excel(writer, sheet_name='Line Tension', index=False)
    print(f"All data saved to: {data_save_path}")
    
    
def main():
    data_path, tif_path = getFiles()
    save_path, data_save_path = getSaveFiles(tif_path)
    lineTension(data_path, tif_path, save_path, data_save_path)
    
    
    

if __name__ == "__main__":
    main()
    

    
    


