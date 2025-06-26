from skimage import io
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, disk
from skimage.measure import label, regionprops
from skimage.draw import ellipse_perimeter, ellipse, line
import pandas as pd
import numpy as np

''' [fitEllipses.py] Last Updated: 6/6/2025 by Myles Koppelman '''
 
def fitEllipse(data_path, tif_path, head):
    """
    Fits ellipses to the identified domains in a multi-page TIFF stack and computes their major and minor axes lengths,
    as well as the area of the region within the ellipse that overlaps with the black (zero-valued) domain.

    Parameters:
    ----------
    data_path : str
        Path to the Excel file containing region properties for each frame and particle.
    tif_path : str
        Path to the multi-page TIFF stack to analyze.
    head : int
        Index of the specific domain ('head') to track (1-based indexing).

    Returns:
    -------
    tuple
        - areas : list of int
            List of calculated areas (number of black pixels within the ellipse) for the selected 'head' across frames.
        - majors : list of float
            List of major axis lengths of the fitted ellipses for the 'head'.
        - minors : list of float
            List of minor axis lengths of the fitted ellipses for the 'head'.

    Notes:
    -----
    - The ellipse parameters (`Centroid_X`, `Centroid_Y`, `Major_Axis`, `Minor_Axis`, `Orientation`) are extracted
      from the Excel file.
    - Ellipses are drawn using these parameters, and their overlap with the black domains in each frame is computed.
    - Only the ellipses corresponding to the specified 'head' index are included in the output lists.
    - Useful for measuring domain shape evolution and calculating line tension in monolayer systems.
    """
    
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
    df_all = pd.concat(dfs, ignore_index=True)

    tif_stack = io.imread(tif_path).astype(np.uint8)

    areas = []
    majors = []
    minors = []

    for _, row in df_all.iterrows():
        idx = int(row["Index"])
        frame = int(row["Frame"])
        cy = row["Centroid_Y"]
        cx = row["Centroid_X"]
        a = row["Major_Axis"] / 2
        b = row["Minor_Axis"] / 2
        orientation = row["Orientation"]

        if 0 <= frame < tif_stack.shape[0]:
            img = tif_stack[frame]
            orientation = -(orientation + np.pi / 2) # correct orientation
            rr2, cc2 = ellipse(
                int(round(cy)),
                int(round(cx)),
                int(round(b)),
                int(round(a)),
                rotation=orientation,
                shape=img.shape
            )
            ellipse_mask = np.zeros_like(img, dtype=bool)
            ellipse_mask[rr2, cc2] = True
            
            x0_major = cx - (a * np.cos(orientation)) # Calculate endpoints of major axis
            y0_major = cy - (a * np.sin(orientation))
            x1_major = cx + (a * np.cos(orientation))
            y1_major = cy + (a * np.sin(orientation))
            
            major = np.sqrt((x1_major - x0_major)**2 + (y1_major - y0_major)**2) # major axis length
            
            minor_orientation = orientation + np.pi / 2
            x0_minor = cx - (b * np.cos(minor_orientation)) # Calculate endpoints of minor axis
            y0_minor = cy - (b * np.sin(minor_orientation))
            x1_minor = cx + (b * np.cos(minor_orientation))
            y1_minor = cy + (b * np.sin(minor_orientation))
            
            minor = np.sqrt((x1_minor - x0_minor)**2 + (y1_minor - y0_minor)**2) # minor axis length

            area = np.sum((img == 0) & ellipse_mask) # area = all black pixels inside ellpise mask
            if head == idx:
                areas.append(area)
                majors.append(major)
                minors.append(minor)
                
    return areas, majors, minors
            
          