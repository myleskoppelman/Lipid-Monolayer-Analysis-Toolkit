import numpy as np
import  os
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

''' [arcLength.py] Last Updated: 6/6/2025 by Myles Koppelman '''

def arcLength(tif_path):
    """
    Calculates the arc length of black (zero-valued) domains in each frame of a multi-page TIFF stack
    by fitting a polynomial curve to the domain coordinates.

    Parameters:
    ----------
    tif_path : str
        Path to the multi-page TIFF file to analyze.

    Returns:
    -------
    tuple
        A tuple containing:
        - arc_lengths : list of float
            List of calculated arc lengths (one per frame).
        - poly_funcs : list of numpy.poly1d
            List of polynomial functions fitted to the domain boundaries (one per frame).

    Notes:
    -----
    - For each frame:
        - The black pixels (with value 0) are identified.
        - A second-degree polynomial is fitted to the (x, y) coordinates of the black pixels.
        - A PDF report (`_ARCLEN.pdf`) is saved in the same folder as the TIFF file, showing:
            - The domain points (scatter plot).
            - The fitted polynomial curve.
    - This function is useful for quantifying the dynamic behavior of domains in monolayers or similar structures.
    """
    
    path, filename = os.path.split(tif_path)
    name, _ = os.path.splitext(filename)
    
    tif_stack = io.imread(tif_path).astype(np.uint8)
    
    if tif_stack.ndim == 2:
        tif_stack = tif_stack[np.newaxis, :, :]

    arc_lengths = []
    poly_funcs = []


    with PdfPages(os.path.join(path, f"{name}_ARCLEN.pdf")) as pdf:
        for i, img in enumerate(tqdm(tif_stack, desc="Fitting Arc Length")):
            frame_array = np.array(img)
            rows, cols = np.where(frame_array == 0) # gather rows, cols with black pixels
            
            sorted_indices = np.argsort(cols)
            x_sorted = cols[sorted_indices]
            y_sorted = rows[sorted_indices]
            
            degree = 2
            coeffs = np.polyfit(x_sorted, y_sorted, degree) # fit curve over black pixels
            poly_func = np.poly1d(coeffs) # get coefficients of curve


            x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 1000) # smooth curve
            y_fine = poly_func(x_fine)

            dx = np.diff(x_fine)
            dy = np.diff(y_fine)
            arc_length_poly = np.sum(np.sqrt(dx**2 + dy**2)) # calculate arc length


            plt.scatter(x_sorted, y_sorted, s=5, label='Domains') #
            plt.plot(x_fine, y_fine, 'r-', label='Arc Length')
            plt.gca().invert_yaxis()
            plt.legend()
            plt.title(f"Frame {i}")
            pdf.savefig()  
            plt.close()
            
            arc_lengths.append(arc_length_poly)
            poly_funcs.append(poly_func) 
        
    return arc_lengths, poly_funcs