import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy.fft import fft, ifft,fftfreq
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

sns.set_theme(style = "whitegrid")
sns.set_context("talk", font_scale = 1.5)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['axes.axisbelow'] = True
matplotlib.rcParams['figure.figsize'] = [10, 7]

from utils import *
from isolateAllDomains import *


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
        x_end   = min(bbx + bbw + padding, img_width)
        y_end   = min(bby + bbh + padding, img_height)

        try:
            frame_slice = tif_stack[frame][y_start:y_end, x_start:x_end]
        except:
            frame_slice = tif_stack[y_start:y_end, x_start:x_end]

        yy, xx = np.where(frame_slice == domain_color) # Gather 'domain' colored pixels
        domain_pixels = np.column_stack((yy + y_start - centroid_y, xx + x_start - centroid_x)) 

        coords.append(domain_pixels)

    return coords


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5) -> np.ndarray:
    """
    High-pass filter a 1D array (radial distances along a contour).

    Parameters:
        data: 1D array of distances along the contour
        cutoff: cutoff frequency in cycles per contour (normalized by sample_rate)
        sample_rate: number of points along the contour
        poles: filter order

    Returns:
        filtered_data: high-pass filtered array
    """
    # Design high-pass Butterworth filter
    # if len(data) < 18:  # not enough points for filtfilt
    #     return data  
    # if cutoff >= sample_rate/2:
    #     cutoff = sample_rate/2 - 1e-3 
    sos = signal.butter(poles, cutoff, btype='highpass', fs=sample_rate, output='sos')
    
    # Apply zero-phase filtering
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data



def analyzeContour(contours, frame, scale, sheet_name):
    
    freqs = []
    shape = []
    records = []
    dfs = []
    graphs = []
    

        
    for idx, contour in enumerate(contours): 
        
        if len(contour) < 2:  
            continue
        
        centroid = np.mean(contour, axis=0)
        distances = np.linalg.norm(contour - centroid, axis=1)
        diffs = gaussian_filter(distances, 2)

        filtered = highpass(diffs, cutoff=8, sample_rate=len(diffs))


        length = 0
        for k in range(len(contour) - 1):
            p1 = np.array([contour[k, 0], contour[k, 1]])
            p2 = np.array([contour[k+1, 0], contour[k+1, 1]])
            length += np.linalg.norm(p1 - p2) * scale
        xlen = np.linspace(0, length, len(contour))
        shape.append(length)

        # FFT of radial fluctuations
        N = len(contour)
        T = 1 / N
        yf = fft(filtered)
        xf = fftfreq(N, T)[:N // 2]
        
        spectrum = 2.0 / N * np.abs(yf[0:N//2])
        ind = np.argmax(spectrum)
        dominant_freq = xf[ind]
        dominant_magnitude = spectrum[ind]
        

        time_series = filtered
        time = np.arange(0, len(time_series), 1)  
        sine_wave = np.sin(2 * np.pi * dominant_freq * (time+8) / len(time_series))*2
        

        freqs.append(xf[ind])
        
        records.append({
            "Contour": idx,
            "Length (um)": length,
            "Dominant Freq (arb)": dominant_freq,
            "Dominant Magnitude": dominant_magnitude,
        })
        
        graphs.append({
            "Contour": idx,
            "Raw Distances": np.roll(distances,200),
            "Scaled Distances": np.roll(filtered, 200),
            "Frequency Magnitudes": 2.0/N * np.abs(yf[0:N//2]),
            "Sine Wave": sine_wave,
            "x": xlen,
            "frame": frame
        })
        

    freqs = np.array(freqs)
    df = pd.DataFrame(records)

    return freqs, shape, df, graphs
    
    
    




def main():
    print("\n\n\nRunning domainWavelength.py. Please respond to all GUI prompts...\n\n")
    
    _ = easygui.buttonbox(
        msg="domainWavelength.py\n\n    !!!WARNING!!!\nThis program will save a very large PDF file of graphs if your data has a lot of domains...",
        title="Settings",
        choices=["Continue"]
    )

    
    original_tif_path = getTif("Select Binary .tif File")
    xlsx_path = getXlsx("Select Particle Tracking Data")
    xlsx_save = saveXlsx("Select location to save output data", "_DATA", original_tif_path)
    pdf_path = savePdf("Select location to save output graph data", "_GRAPHS", original_tif_path)

    
    scale = easygui.multenterbox(
        msg="Enter the number of microns per pixel:",
        title="Settings",
        fields=["Scale (microns/pixel):"],
        values=["0.222222"] 
    )
    
    if scale is None:
        raise Exception("User canceled the input dialog.")

    try:
        scale = float(scale[0])
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")


    tif_path = isolateDomains(xlsx_path, original_tif_path)
    
    freqs = []
    shapes = []
    sheets = {}
    graphs = []
    
    tif_stack = io.imread(tif_path).astype(np.uint8)
    

    num_frames = 1 if tif_stack.ndim == 2 else tif_stack.shape[0]
    

    i = 0

    for frame in tqdm(range(num_frames), desc="Processing Frames"):
        i += 1
        if tif_stack.ndim == 2:
            frame = tif_stack  # single image
        else:
            frame = tif_stack[frame]  
            
        contour = measure.find_contours(frame<50, 0)

        
        sheet_name = f"Frame_{i}"

            
    
        freq, shape, df, graph  = analyzeContour(contour, i, scale, sheet_name)
        
        
        
        freqs.extend(freq)
        shapes.extend(shape)
        
        
        if not df.empty:
            sheets[sheet_name] = df
            graphs.extend(graph) 
            
        
    shapes = np.array(shapes)
    freqs = np.array(freqs)   
                

    with PdfPages(pdf_path) as pdf:

        plt.figure(constrained_layout=True)
        sns.histplot((shapes/freqs)/1.4)
        plt.title(f"Median Wavelength: {np.median((shapes/freqs)/1.4)}")
        plt.xlabel("Wavelength (um)")
        pdf.savefig()
        plt.close()
        
        
        for g in tqdm(graphs, desc="Saving plots", unit="graph"):
            contour_id = g["Contour"]
            frame_id = g["frame"]

            # Plot Raw Distances
            plt.figure()
            plt.plot(g["x"], g["Raw Distances"])
            plt.xlabel("Contour Length (um)")
            plt.ylabel("Distance to Centroid")
            plt.title(f"Frame {frame_id} - Contour {contour_id} - Raw Distances")
            pdf.savefig()
            plt.close()

            # Plot Scaled Distances
            plt.figure()
            plt.plot(g["x"], g["Scaled Distances"])
            plt.xlabel("Contour Length (um)")
            plt.ylabel("Filtered Distance")
            plt.title(f"Frame {frame_id} - Contour {contour_id} - Scaled Distances")
            pdf.savefig()
            plt.close()

            # Plot Frequency Magnitudes
            plt.figure()
            plt.plot(g["Frequency Magnitudes"], '-o')
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.title(f"Frame {frame_id} - Contour {contour_id} - Frequency Magnitudes")
            pdf.savefig()
            plt.close()

            # Plot Dominant Frequency Overlay
            plt.figure()
            plt.plot(g["x"], g["Scaled Distances"], label="Filtered Distance")
            plt.plot(g["x"], g["Sine Wave"], '--', label="Dominant Frequency")
            plt.ylim((-5, 5))
            plt.xlabel("Contour Length (um)")
            plt.ylabel("Distance to Centroid")
            plt.title(f"Frame {frame_id} - Contour {contour_id} - Dominant Frequency")
            plt.legend()
            pdf.savefig()
            plt.close()

    print(f"Saved all plots to {pdf_path}")
            
        
    with pd.ExcelWriter(xlsx_save, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df = pd.DataFrame(df)
            df.to_excel(writer, sheet_name=sheet_name, index=False)





    
    
        
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()
    