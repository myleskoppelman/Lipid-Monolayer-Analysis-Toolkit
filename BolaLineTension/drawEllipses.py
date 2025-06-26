import tifffile
from skimage import io, draw
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.draw import ellipse_perimeter, ellipse, line
import easygui
import os
import pandas as pd
import numpy as np

''' [drawEllipses.py] Last Updated: 6/6/2025 by Myles Koppelman '''

def getFiles() -> tuple[str, str]:
    """
    [tagParticles.py] Last Updated: 5/30/2025 by Myles Koppelman
    
    
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
    
    
    path, filename = os.path.split(tif_path)
    nm, _ = os.path.splitext(filename)
    
    
    save_path = easygui.filesavebox(
        msg="Save Output .tif File",
        default=os.path.join(path, f"{nm}_ELLIPSE.tif"),
        filetypes=["*.tif"]
    )
    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.tif'):
        save_path += '.tif'
    
    return data_path, tif_path, save_path

    
     
def drawEllipses(data_path, tif_path, save_path, poly_funcs=None):
    """
    Draws ellipses, major and minor axes, and optionally polynomial fits onto each frame of a multi-page TIFF stack,
    and saves the annotated images as a new RGB TIFF stack.

    Parameters:
    ----------
    data_path : str
        Path to the Excel file containing region properties for each frame and particle.
    tif_path : str
        Path to the multi-page grayscale TIFF stack.
    save_path : str
        Path to save the output RGB TIFF stack with ellipses and optional polynomial fits overlaid.
    poly_funcs : list of callable, optional
        List of polynomial functions (as returned by `np.poly1d`) for each frame, representing fits to domain contours.
        If provided, these polynomials will be drawn on the corresponding frames in magenta.

    Notes:
    -----
    - The ellipses are drawn based on properties from the Excel file:
      centroid coordinates (`Centroid_X`, `Centroid_Y`), major/minor axis lengths, and orientation.
    - Major axes are drawn in green, minor axes in blue, and ellipse perimeters in red.
    - If polynomial fits are provided, they are drawn in magenta.
    - The background grayscale image is rendered in black and white, with the RGB overlays on top.
    - The final RGB stack is saved using `tifffile.imwrite` with photometric interpretation set to `'rgb'`.

    Example Usage:
    --------------
    drawEllipses('data.xlsx', 'input.tif', 'output_ellipses.tif', poly_funcs=arc_polynomials)
    """
    
    
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)

    tif_stack = io.imread(tif_path).astype(np.uint8)
    output_stack_rgb = np.full((*tif_stack.shape, 3), fill_value=255, dtype=np.uint8)
    
    for _, row in df_all.iterrows():
        frame = int(row["Frame"])
        if 0 <= frame < tif_stack.shape[0]:
            img = tif_stack[frame]
            binary = img == 0
            output_stack_rgb[frame][binary] = [0, 0, 0]

    for _, row in df_all.iterrows():
        frame = int(row["Frame"])
        cy = row["Centroid_Y"]
        cx = row["Centroid_X"]
        a = row["Major_Axis"] / 2
        b = row["Minor_Axis"] / 2
        orientation = row["Orientation"]

        if 0 <= frame < tif_stack.shape[0]:
            img = tif_stack[frame]

            orientation = -(orientation + np.pi / 2)

            rr, cc = ellipse_perimeter(
                int(round(cy)),
                int(round(cx)),
                int(round(b)),
                int(round(a)),
                orientation=orientation,
                shape=img.shape
            )

            # Major axis
            x0_major = cx - (a * np.cos(orientation))
            y0_major = cy - (a * np.sin(orientation))
            x1_major = cx + (a * np.cos(orientation))
            y1_major = cy + (a * np.sin(orientation))
            rr_major, cc_major = line(int(round(y0_major)), int(round(x0_major)), int(round(y1_major)), int(round(x1_major)))

            # Minor axis
            minor_orientation = orientation + np.pi / 2
            x0_minor = cx - (b * np.cos(minor_orientation))
            y0_minor = cy - (b * np.sin(minor_orientation))
            x1_minor = cx + (b * np.cos(minor_orientation))
            y1_minor = cy + (b * np.sin(minor_orientation))
            rr_minor, cc_minor = line(int(round(y0_minor)), int(round(x0_minor)), int(round(y1_minor)), int(round(x1_minor)))

            # Clip to image boundaries
            rr_major = np.clip(rr_major, 0, img.shape[0]-1)
            cc_major = np.clip(cc_major, 0, img.shape[1]-1)
            rr_minor = np.clip(rr_minor, 0, img.shape[0]-1)
            cc_minor = np.clip(cc_minor, 0, img.shape[1]-1)

            # Draw in RGB stack
            output_stack_rgb[frame, rr, cc] = [255, 0, 0]       # Red ellipse perimeter
            output_stack_rgb[frame, rr_major, cc_major] = [0, 255, 0]  # Green major axis
            output_stack_rgb[frame, rr_minor, cc_minor] = [0, 0, 255]  # Blue minor axis

    if poly_funcs is not None:
        for frame, poly_func in enumerate(poly_funcs):
            if frame >= tif_stack.shape[0]:
                continue
 

            img = tif_stack[frame]
            x_vals = np.linspace(0, img.shape[1] - 1, img.shape[1])
            y_vals = poly_func(x_vals)
            y_vals = np.round(y_vals).astype(int)

            valid = (y_vals >= 0) & (y_vals < img.shape[0])
            x_valid = x_vals[valid].astype(int)
            y_valid = y_vals[valid]

            output_stack_rgb[frame, y_valid, x_valid] = [255, 0, 255]

    tifffile.imwrite(save_path, output_stack_rgb, photometric='rgb')
    print(f"Saved ellipses and polynomials to: {save_path}")





def main():
    data, tif_path, save_path = getFiles()
    drawEllipses(data, tif_path, save_path)
    
    
if __name__ == "__main__":
    main()
    