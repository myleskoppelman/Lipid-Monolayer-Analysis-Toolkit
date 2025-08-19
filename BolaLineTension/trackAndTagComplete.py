import easygui, os, importlib, tifffile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, remove_small_objects
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tifffile import imwrite
from typing import Tuple

''' [trackAndTag.py] Last Updated: 7/11/2025 by Myles Koppelman

This file has all the code necessary to track and tag domains from a multipage tif in one single file.

'''


def getOptions(s1, s2, s3): 
    '''
    [getOptions.py] Last Updated: 5/30/2025 Myles Koppelman
    
    
    Prompt the user for various input options related to particle tracking data processing.

    This function uses easygui dialogs to gather user inputs for:
      - Selecting a binary/raw image sequence (.tif) or a MATLAB domain file (.mat).
      - Setting initial data analysis parameters such as minimum pixel area, minimum frames, 
        maximum area variation, maximum eccentricity, and threshold factor.
      - Setting initial tracking parameters including maximum area change, maximum allowed movement 
        (in pixels), and maximum frame skip.
      - Setting iterative tracking parameters for refining tracking with similar options as initial tracking.
      - Selecting domain color (Dark or Light).
      - Choosing thresholding method (Normal or Adaptive) if the input image is not binary.
      - If Adaptive thresholding is selected, specifying the number of horizontal and vertical slices.
      - Options to save binary converted image files and particle output data files.

    The function also checks that slicing dimensions evenly divide the image dimensions when using adaptive thresholding.

    Returns:
        list: A list of options and parameters including:
            - file_path (str): Path to the selected input image or .mat file.
            - binary (bool): True if input image is binary (pixel values {0,1} or {0,255}), else False.
            - dc (int): Domain color flag, 0 for Dark, 1 for Light.
            - max_eccentricity (float): Maximum eccentricity allowed for particle filtering.
            - min_area (int): Minimum pixel area for particle filtering.
            - min_frames (int): Minimum number of frames a particle must appear in.
            - max_area_variation (float): Maximum area variation allowed in filtering.
            - max_areachange (float): Max area change allowed during initial tracking.
            - max_movement (int): Max allowed movement in pixels during initial tracking.
            - max_frameskip (int): Max frames skipped during initial tracking.
            - max_areachange2 (float): Max area change allowed during iterative tracking.
            - max_movement2 (int): Max allowed movement in pixels during iterative tracking.
            - max_frameskip2 (int): Max frames skipped during iterative tracking.
            - write_binary_img (bool): Whether to save the binary image output.
            - at (bool): Whether adaptive thresholding is used.
            - threshold_factor (float): Threshold factor for image processing.
            - (horz_slices, vert_slices) (tuple): Number of horizontal and vertical slices for adaptive thresholding.
            - binary_save_path (str or None): Path to save binary image file if selected.
            - data_save_path (str): Path to save tracked particle data file.

    Raises:
        Exception: If the user cancels any of the file selection or input dialogs.
        ValueError: If numerical inputs are invalid or adaptive slice dimensions do not evenly divide the image size.

    Notes:
        - This function expects image files to be grayscale or binary images.
        - The user is prompted multiple times for settings that impact initial processing and tracking parameters.
        - Adaptive thresholding requires careful selection of slice counts that exactly divide the image dimensions.
    '''
    
    # Initialize parameters
    horz_slices = 4 
    vert_slices = 8
    at = False
    data_save_path = None
    
    
    # ---------------------------------------------------------------------------------------------------------
    # Select Input File
    # ---------------------------------------------------------------------------------------------------------
    file_path = easygui.fileopenbox(
        msg="Select binary/Raw Image Sequence",
        default="Data#",
        filetypes=["*.tif"]
    )
    if not file_path:
        raise Exception("No file selected.")


    p, filename = os.path.split(file_path)
    nm, _ = os.path.splitext(filename)

    
    # ---------------------------------------------------------------------------------------------------------
    # Settings for Data Processing
    # ---------------------------------------------------------------------------------------------------------
    settings = easygui.multenterbox(
        msg="Enter Data Analysis Settings: These settings affect the initial processing of the data, but do not affect tracking...",
        title="Settings",
        fields=["Minimum Domain Area", "Maximum Domain Area","Minimum Frames", "Maximum Area Variation", "Maximum Eccentricity", "Threshold Factor"],
        values=s1 # Change these to alter inital settings
    )
    if settings is None:
        raise Exception("User canceled the input dialog.")

    try:
        min_area = int(settings[0])
        max_area = int(settings[1])
        min_frames = int(settings[2])
        max_area_variation = float(settings[3])
        max_eccentricity = float(settings[4])
        threshold_factor = float(settings[5])
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")
    
    
    
    # ---------------------------------------------------------------------------------------------------------
    # Settings for Initail Tracking
    # ---------------------------------------------------------------------------------------------------------
    settings2 = easygui.multenterbox(
        msg="Enter Initial Tracking Settings: These settings affect the intial tracking iteration. Generally you want these values flexible for the first track but tighten them during iterative tracking...",
        title="Settings",
        fields=["Maximum Area Variation", "Maximum Allowed Movement(Pixels)", "Maximum Frameskip"],
        values=s2  # Change these to alter inital settings
    )
    if settings2 is None:
        raise Exception("User canceled the input dialog.")

    try:
        max_areachange = float(settings2[0])
        max_movement = int(settings2[1])
        max_frameskip = int(settings2[2])
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")
    
    
    
    
    # ---------------------------------------------------------------------------------------------------------
    # Settings for Iterative Tracking
    # ---------------------------------------------------------------------------------------------------------
    settings3 = easygui.multenterbox(
        msg="Enter Iterative Tracking Settings: These settings affect the iterative tracking. Iterative tracking analyzes the data from the previous tracking cycle to model the average displacement of the domains. For this cycle you want to keep the allowed movement low because the program already is guessing where the particle should be...",
        title="Settings",
        fields=["Maximum Area Variation", "Maximum Allowed Movement(Pixels)", "Maximum Frameskip"],
        values=s3  # Change these to alter inital settings
    )
    if settings3 is None:
        raise Exception("User canceled the input dialog.")

    try:
        max_areachange2 = float(settings3[0])
        max_movement2 = int(settings3[1])
        max_frameskip2 = int(settings3[2])
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")

    
    
    # ---------------------------------------------------------------------------------------------------------
    # Gather # frames and Image Size
    # ---------------------------------------------------------------------------------------------------------
    img = Image.open(file_path)
    img_array = np.array(img.convert('L'))
    width, height = img.size
    img.close()
    
    unique_vals = set(img_array.flatten())
    binary = unique_vals in ({0, 1}, {0, 255}) # Check if Binary
    
    if not binary:
        bin_tif_path = easygui.filesavebox(
            msg="Save Output Binary .tif File",
            default=os.path.join(p, f"{nm}_BIN.tif"),
            filetypes=["*.tif"]
        )
        if not bin_tif_path:
            raise Exception("No output file selected.")
        
        if bin_tif_path is not None and not bin_tif_path.lower().endswith('.tif'):
            bin_tif_path += '.tif'
            
    else:
        bin_tif_path = None


    # ---------------------------------------------------------------------------------------------------------
    # Ask Domain Color
    # ---------------------------------------------------------------------------------------------------------
    domain_color = easygui.buttonbox(
        msg="Domain Color",
        title="Settings",
        choices=["Dark", "Light"]
    )

    dc = 0 if domain_color == "Dark" else 1
    
    # ---------------------------------------------------------------------------------------------------------
    # Ask to consider domains touching the border
    # ---------------------------------------------------------------------------------------------------------
    border_ = easygui.buttonbox(
        msg="Consider Domains touching the border?",
        title="Settings",
        choices=["Yes", "No"]
    )

    border = False if border_ == "No" else True
    
    # ---------------------------------------------------------------------------------------------------------
    # Ask iterative tracking
    # ---------------------------------------------------------------------------------------------------------
    iterative = easygui.buttonbox(
        msg="Conduct Iterative Tracking? (select no if you are tracking only one frame)",
        title="Settings",
        choices=["Yes", "No"]
    )

    iterative_tracking = False if iterative == "No" else True
    
    
    
    # ---------------------------------------------------------------------------------------------------------
    #  Ask for threshold settings and wether to save binary images
    # ---------------------------------------------------------------------------------------------------------
    if not binary:
        adaptive_thresh = easygui.buttonbox(
            msg="Select Threshold",
            title="Settings",
            choices=["Normal", "Adaptive (not working)"]
        )
        


        # Prompt User for number of horizontal and vertical slices for adaptive threshold 
        if adaptive_thresh == "Adaptive":
            at = True
            valid = False
            while not valid:
                slices = easygui.multenterbox(
                    msg="Enter slicing settings",
                    title="Settings",
                    fields=["Horizontal Slices", "Vertical Slices"],
                    values=["4", "8"]
                )
            
                if slices is None:
                    raise ValueError("Invalid input. Make sure to enter numbers.")
                
                try:
                    horz_slices = int(slices[0])
                    vert_slices = int(slices[1])
                except ValueError:
                    easygui.msgbox("Inputs must be integers.", "Error")
                    continue

                h1 = width / horz_slices
                v1 = height / vert_slices
                h2 = round(h1)
                v2 = round(v1)

                if h1 == h2 and v1 == v2:
                    valid = True
                else:
                    easygui.msgbox("Slice values must evenly divide the image dimensions. Try again.", "Invalid Input")
    else:
        at = False
        
        
    # ---------------------------------------------------------------------------------------------------------
    #  Edit save file names
    # ---------------------------------------------------------------------------------------------------------    
    if dc == 0:
        nm = nm + "_DARK"
    else:
        nm = nm + "_LIGHT"
          
    
    # ---------------------------------------------------------------------------------------------------------
    #  Ask location to save files
    # ---------------------------------------------------------------------------------------------------------


    data_save_path = easygui.filesavebox(
        msg="Save Output .xlsx File",
        default=os.path.join(p, f"{nm}_TRACKED.xlsx"),
        filetypes=["*.xlsx"]
    )
    if not data_save_path:
        raise Exception("No output file selected.")
    
    if data_save_path is not None and not data_save_path.lower().endswith('.xlsx'):
        data_save_path += '.xlsx'
        
    
    tagged_tif_path = easygui.filesavebox(
        msg="Save Output .tif File",
        default=os.path.join(p, f"{nm}_TAGGED.tif"),
        filetypes=["*.tif"]
    )
    if not tagged_tif_path:
        raise Exception("No output file selected.")
    
    if tagged_tif_path is not None and not tagged_tif_path.lower().endswith('.tif'):
        tagged_tif_path += '.tif'


    print("Options Complete...")
    return [
        file_path,
        binary,
        dc,
        border,
        iterative_tracking,
        max_eccentricity,
        min_area,
        max_area,
        min_frames, 
        max_area_variation,
        max_areachange,
        max_movement,
        max_frameskip,
        max_areachange2,
        max_movement2,
        max_frameskip2,
        at,
        threshold_factor,
        (horz_slices, vert_slices),

        data_save_path,
        bin_tif_path,
        tagged_tif_path
    ]


def thresholdGui(frames, out_path):
    """
    Display an interactive GUI to adjust thresholding on a stack of grayscale image frames, 
    apply the selected threshold to all frames, and save the result as a binary TIFF.

    Parameters
    ----------
    frames : list of ndarray
        A list of 2D NumPy arrays representing grayscale image frames.
    out_path : str
        The file path where the binary thresholded TIFF stack will be saved.

    Returns
    -------
    str
        The file path where the binary thresholded TIFF has been saved.

    Raises
    ------
    Exception
        If no output path is provided or saving fails.
    
    Notes
    -----
    • The GUI includes sliders to adjust black point, white point, and frame selection.
    • A button applies the selected threshold to all frames and saves the result as a multi-page TIFF.
    • Pixels within the threshold range are highlighted in red for preview.
    """
    current_frame = 0
    frame = frames[current_frame]

    init_low = np.percentile(frame, 5)
    init_high = np.percentile(frame, 95)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    
    def thresholdOverlay(frame, low, high):
        rgb = np.stack([frame]*3, axis=-1)
        mask = (frame >= low) & (frame <= high)
        rgb[mask] = [255, 0, 0] 
        return rgb

    img_disp = ax.imshow(thresholdOverlay(frame, init_low, init_high))
    ax.set_title(f"Frame {current_frame + 1}/{len(frames)}")


    ax_low = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_high = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_frame = plt.axes([0.25, 0.1, 0.65, 0.03])

    s_low = Slider(ax_low, 'Black Pt', 0, 255, valinit=init_low, valstep=1)
    s_high = Slider(ax_high, 'White Pt', 0, 255, valinit=init_high, valstep=1)
    s_frame = Slider(ax_frame, 'Frame', 0, len(frames)-1, valinit=0, valstep=1)


    ax_button = plt.axes([0.4, 0.025, 0.2, 0.05])
    save_button = Button(ax_button, 'Apply & Save')


    def update(val):
        f = int(s_frame.val)
        low = int(s_low.val)
        high = int(s_high.val)
        img_disp.set_data(thresholdOverlay(frames[f], low, high))
        ax.set_title(f"Frame {f + 1}/{len(frames)}")
        fig.canvas.draw_idle()

    s_low.on_changed(update)
    s_high.on_changed(update)
    s_frame.on_changed(update)
    
    def applyThreshold(frame, low, high):
        return ((frame >= low) & (frame <= high)).astype(np.uint8) * 255


    def save(event):
        print("Applying threshold and saving...")
        low = int(s_low.val)
        high = int(s_high.val)
        binary_stack = np.array([applyThreshold(f, low, high) for f in frames], dtype=np.uint8)
        imwrite(out_path, binary_stack, photometric='minisblack')
        print(f"Saved to {os.path.abspath(out_path)}")
        plt.close()
        return

    save_button.on_clicked(save)

    plt.show()
    
    if out_path is not None:
        return out_path
    else:
        raise Exception("Binary Threshold Failed...")


def threshold(tif_path: str, tif_save_path: str) -> Tuple[str, str]:
    """
    Load a multi-frame TIFF file, launch an interactive thresholding GUI, 
    and save the thresholded binary image stack to a new TIFF file.

    Parameters
    ----------
    tif_path : str
        The file path of the input multi-frame TIFF to be thresholded.
    tif_save_path : str
        The file path where the thresholded binary TIFF will be saved.

    Returns
    -------
    tuple of str
        A tuple containing the input TIFF path and the saved output TIFF path.

    Notes
    -----
    • The function reads all frames from the input TIFF, displays the GUI 
      for threshold selection, and applies the chosen threshold to each frame.
    • The result is saved as an 8-bit grayscale binary TIFF with pixels either 0 or 255.
    """
    img = Image.open(tif_path)
    frames = [np.array(frame.convert('L')) for frame in ImageSequence.Iterator(img)]
    img.close()
    tif_save_path = thresholdGui(frames, tif_save_path)
    return tif_path, tif_save_path


def preProcess(file_path, bin_tif_path, is_bin, domain_color, border, max_eccentricity, min_area, adaptive_thresh, threshold_factor, slices):
    '''
    [preProcessFile.py] Last Updated: 5/30/2025 by Myles Koppelman
    
    
    Process an input image sequence for particle detection via thresholding and particle analysis.

    This function reads a multi-frame image (e.g., TIFF sequence) and performs the following:
      - Converts frames to grayscale arrays.
      - Applies thresholding (either normal Otsu or adaptive thresholding in spatial slices) to segment particles.
      - Inverts the binary mask if the domain color is dark.
      - Cleans the binary mask by morphological operations (opening, hole filling, border clearing).
      - Removes small objects below a minimum size threshold.
      - Labels connected components to identify individual particles/domains.
      - Extracts particle properties such as area, centroid coordinates, eccentricity, and bounding box.
      - Filters out particles with eccentricity greater than the specified maximum eccentricity threshold.
      - Optionally saves the processed binary frames as an output TIFF file.

    Args:
        file_path (str): Path to the input multi-frame image file (.tif).
        bin (bool): Flag indicating if input is binary (True) or grayscale/raw (False).
        domain_color (int): 0 for dark domain (invert mask), 1 for light domain.
        me (float): Maximum eccentricity threshold for particle filtering.
        min_area (int): Minimum pixel size for particle retention after thresholding.
        write_bin (bool): Whether to save the processed binary images.
        adaptive_thresh (int): 0 for normal Otsu threshold, 1 for adaptive thresholding by image slicing.
        threshold_factor (float): Scaling factor applied to Otsu threshold values.
        slices (tuple): (horz_slices, vert_slices) specifying number of slices for adaptive thresholding.
        bin_save_path (str): File path to save the binary output TIFF if write_bin is True.

    Returns:
        np.ndarray: An array of detected particle properties with columns:
            [Area, CentroidX, CentroidY, FrameIndex, Eccentricity, BoundingBoxX, BoundingBoxY, BoundingBoxWidth, BoundingBoxHeight]

    Raises:
        None explicitly, but file opening errors or invalid image formats may propagate.

    Notes:
        - Adaptive thresholding divides each frame into spatial blocks and applies Otsu thresholding locally.
        - Morphological cleaning steps help remove noise and fill particle shapes.
        - Filtering by eccentricity removes elongated or irregularly shaped particles beyond the threshold.
        - Prints a warning if no particles are detected in a frame, suggesting domain color inversion may be needed.
    '''
    
    horz_slices = slices[0]
    vert_slices = slices[1]
    
    if not is_bin:
        if bin_tif_path is None:
            raise Exception("No Binary .tif save path...")
        _, bin_tif_path = threshold(file_path, bin_tif_path) 
    else:
        is_bin = True
        bin_tif_path = file_path
        

    img = Image.open(bin_tif_path)
    n_frames = [np.array(frame.copy().convert('L')) for frame in ImageSequence.Iterator(img)]
    img.close()
    
    processed_frames = []
    output_data = []
    eccentricities = []
    ne = 0  # Counter for frames with no/few domains


    for n , im in enumerate(tqdm(n_frames, desc="Thresholding and Analyzing Particles")):
    # ---------------------------------------------------------------------------------------------------------
    # THRESHOLDING
    # ---------------------------------------------------------------------------------------------------------
        t = threshold_otsu(im) * threshold_factor 
        binary = im > (t * 0.8)

        if domain_color == 0:
            binary = ~binary
            
        binary = binary_opening(binary)
        # ----------------------- Remove these for unedited binary image ---------------------------
        binary = remove_small_objects(binary, min_area) 
        # binary = binary_fill_holes(binary)
        if not border:
            binary = clear_border(binary)
        # ------------------------------------------------------------------------------------------

        processed_frames.append(img_as_ubyte(~binary))       
        # else:
        #     binary = ~im.astype(bool)

    # ---------------------------------------------------------------------------------------------------------
    # IDENTIFY PARTICLES
    # ---------------------------------------------------------------------------------------------------------
        labeled, num = label(binary, return_num=True, connectivity=2)

        if num > 0:  # [Area, CentroidX, CentroidY, Frame#, Eccentricity, BoundingBoxX, BoundingBoxY, BoundingBoxW, BoundingBoxH]
            props = regionprops(labeled)
            m = np.zeros((len(props), 14))

            m[:, 0] = [p.area for p in props]
            m[:, 1] = [p.centroid[1] for p in props]
            m[:, 2] = [p.centroid[0] for p in props]
            m[:, 3] = n  # Frame index
            m[:, 4] = [p.eccentricity for p in props]

            # Bounding box: p.bbox gives (min_row, min_col, max_row, max_col)
            for i, p in enumerate(props):
                min_row, min_col, max_row, max_col = p.bbox
                m[i, 5] = min_col            # BoundingBoxX (x)
                m[i, 6] = min_row            # BoundingBoxY (y)
                m[i, 7] = max_col - min_col  # BoundingBoxW (width)
                m[i, 8] = max_row - min_row  # BoundingBoxH (height)
                m[i, 9] = p.major_axis_length
                m[i, 10] = p.minor_axis_length
                m[i, 11] = p.orientation  # In radians
                m[i, 12] = 0 if domain_color == 0 else 255
                m[i, 13] = int(p.euler_number < 1) # does the domain have holes

            eccentricities.extend(m[:, 4])
            output_data.extend(m.tolist())

        else:
            print(f"No Domains Found on Frame {n}...")
            ne += 1


    eccentricities = np.array(eccentricities)
    odata = np.array(output_data)
    if eccentricities.size > 0:
        row = np.where(eccentricities > max_eccentricity)[0]
        odata = np.delete(odata, row, axis=0)


    print("Threshold and Analysis Complete...")
    return odata, file_path


def average(particles, n_frames):
    '''
    [averageDisplacement.py] Last Updated: 5/30/2025 by Myles Koppelman 
    
    
    Calculate average frame-to-frame displacements in x and y directions for tracked particles,
    to be used in iterative tracking.

    For each particle trajectory (represented as a numpy array of observations),
    this function computes the displacement vectors between consecutive frames,
    filters out outliers beyond 2 standard deviations from the mean displacement,
    and calculates the average displacement per frame across all particles.

    Args:
        particles (list of np.ndarray): Each element is a 2D array for a single particle,
            where columns correspond to properties including at least:
            - column 1: x position,
            - column 2: y position,
            - column 3: frame index.
        n_frames (int): Total number of frames in the dataset.

    Returns:
        tuple of np.ndarray: (dx, dy), each of length n_frames,
            where dx[i] and dy[i] represent the average displacement in the x and y
            directions, respectively, from frame i to frame i+1.
            Frames with no displacement data return zero.

    Notes:
        - Outliers in displacement are filtered using a 2-standard-deviation cutoff.
        - Displacements are computed as differences in positions between consecutive frames.
        - The function handles cases where no particles move between frames by assigning zero displacement.
    '''
    
    data = []

    for particle in tqdm(particles, desc='Calculating Displacements'):
        frame = particle[1:, 3]
        xdisp = particle[1:, 1] - particle[:-1, 1]
        ydisp = particle[1:, 2] - particle[:-1, 2]
        data.append(np.column_stack((frame, xdisp, ydisp)))

    displacements = np.vstack(data)
    dx = np.zeros(n_frames)
    dy = np.zeros(n_frames)

    for n in tqdm(range(1, n_frames + 1), desc='Calculating Averages'):
        rows = displacements[:, 0] == n
        x_vals = displacements[rows, 1]
        y_vals = displacements[rows, 2]

        if len(x_vals) == 0 or len(y_vals) == 0:
            dx[n-1] = 0
            dy[n-1] = 0
            continue

        def filter_outliers(values):
            mean_val = np.mean(values)
            std_val = np.std(values)
            filtered = values[np.abs(values - mean_val) <= 2 * std_val]
            return filtered

        x_filtered = filter_outliers(x_vals)
        y_filtered = filter_outliers(y_vals)

        dx[n-1] = np.mean(x_filtered) if x_filtered.size > 0 else 0
        dy[n-1] = np.mean(y_filtered) if y_filtered.size > 0 else 0

    return dx, dy


def removeParticles(data, min_area, max_area, max_eccentricity):
    """
    Removes particles from the dataset based on minimum area and maximum eccentricity.

    Parameters:
    - data (np.ndarray): The particle dataset, where the first column represents the area
                         and the fifth column represents the eccentricity.
    - min_area (float): The minimum allowed area for particles to keep.
    - max_eccentricity (float): The maximum allowed eccentricity for particles to keep.

    Returns:
    - np.ndarray: The filtered dataset containing only particles that meet the criteria.
    """
    data = data[data[:, 0] > min_area]
    data = data[data[:, 0] < max_area]
    data = data[data[:, 4] < max_eccentricity]
    print("Successfully removed Particles smaller than min_area or larger than max_eccentricity...")
    return data


def filterParticles(particles, max_area_variation, min_frames):
    """
    Filters out particles based on area variation and minimum number of frames.

    Parameters:
    - particles (list of np.ndarray): A list of particle trajectories, each as a NumPy array.
    - max_area_variation (float): The maximum allowed relative variation in area
                                  for a particle to be considered consistent.
    - min_frames (int): The minimum number of frames a particle must be present in
                        after filtering to be kept.

    Returns:
    - list of np.ndarray: A list of filtered particle trajectories that meet the criteria.
                           Returns an empty list if no particles remain after filtering.
    """
    filtered_particles = []
    
    for particle in tqdm(particles, desc="Removing Unwanted Particles"):
        # areas = particle[:, 0]
        # mean_area = np.mean(areas)
        
        # if mean_area == 0:
        #     continue  # Avoid divide-by-zero

        # area_flux = np.abs(1 - (areas / mean_area))
        # particle_filtered = particle[area_flux <= max_area_variation]

        if particle.shape[0] > min_frames:
            filtered_particles.append(particle)

    if not filtered_particles:
        print("No domains found!")
        return []
    print("Filtering Unwanted Particles Complete...")
    return filtered_particles


def trackParticles(data, max_areachange, max_movement, max_frameskip, xm=None, ym=None):
    """
    Tracks particles across sequential frames, linking detections into trajectories based 
    on area similarity, spatial proximity, and maximum frame gap.

    Parameters:
    ----------
    data : np.ndarray
        The dataset of detected particles. Each row should represent a detection with the 
        following structure: [area, x, y, frame_number, ...].
    max_areachange : float
        The maximum allowed relative change in particle area between detections for linking 
        (expressed as a fraction of the previous area's size).
    max_movement : float
        The maximum allowed distance a particle can move per frame to be considered for 
        linking (in pixels per frame).
    max_frameskip : int
        The maximum number of frames a particle can disappear (be skipped) before the 
        track ends.
    xm : np.ndarray, optional
        The x-drift correction array (one entry per frame) to be applied to account for 
        global frame drift. If not provided, initialized to zeros.
    ym : np.ndarray, optional
        The y-drift correction array (one entry per frame) to be applied to account for 
        global frame drift. If not provided, initialized to zeros.

    Returns:
    -------
    list of np.ndarray
        A list of particle trajectories (tracks). Each trajectory is represented as a NumPy array 
        with the same columns as the input data. Empty trajectories are removed.

    Notes:
    -----
    - The function progressively builds each particle track by finding the next best matching 
      detection in subsequent frames, considering both distance and area consistency.
    - It also applies drift corrections using the provided `xm` and `ym` arrays.
    - The progress of the tracking is displayed using a tqdm progress bar.
    """
    data = data.copy()
    l = data.shape[0]
    particles = [] 
    
    if xm is None or ym is None:
        max_frame = int(np.max(data[:, 3]))
        xm = np.zeros(max_frame + 1)
        ym = np.zeros(max_frame + 1)

    n = -1  # track index
    u = 1   # for progress tracking
    

    with tqdm(total=l, desc="Tracking Particles") as pbar:
        while data.shape[0] > 0:
            n += 1 # Start new track with first detection
            track = [data[0]]
            data = np.delete(data, 0, axis=0)
            
            next_found = False
            while not next_found:
                last_frame = int(track[-1][3])
                
                start_frame = last_frame + 1
                end_frame = last_frame + 1 + max_frameskip
                
                # Find indices of candidate detections in the frame range
                candidate_mask = (data[:, 3] >= start_frame) & (data[:, 3] <= end_frame)
                candidate_indices = np.where(candidate_mask)[0]
                
                ranking = []
                
                for _, idx in enumerate(candidate_indices):
                    candidate = data[idx]
                    framejump = int(candidate[3] - last_frame)
                    
                    # Drift correction sums
                    xest = np.sum(xm[last_frame + 1 : last_frame + 1 + framejump])
                    yest = np.sum(ym[last_frame + 1 : last_frame + 1 + framejump])
                    
                    # Distance with drift correction
                    dist = np.sqrt(
                        (track[-1][1] - candidate[1] + xest) ** 2 +
                        (track[-1][2] - candidate[2] + yest) ** 2
                    )
                    
                    area_diff = abs(track[-1][0] - candidate[0])
                    
                    # Check if candidate meets criteria
                    if dist <= max_movement * framejump and area_diff <= max_areachange * track[-1][0]:
                        ranking.append([idx, dist, area_diff, framejump])
                
                if ranking:
                    ranking = np.array(ranking)
                    if ranking.shape[1] < 7:
                        ranking = np.hstack((ranking, np.zeros((ranking.shape[0], 3))))

                    ranking[:, 4] = ranking[:, 1] / np.sum(ranking[:, 1])  # Displacement Score
                    area_sum = np.sum(ranking[:, 2])
                    if area_sum == 0:
                        ranking[:, 5] = 0
                    else:
                        ranking[:, 5] = ranking[:, 2] / area_sum # Area Score
                    ranking[:, 6] = ranking[:, 3] + ranking[:, 4] + ranking[:, 5]  # Total Score
                    
                    best_idx = np.argmin(ranking[:, 6])
                    chosen_row = int(ranking[best_idx, 0])
                    track.append(data[chosen_row])
                    
                    data = np.delete(data, chosen_row, axis=0)
                else:
                    next_found = True
            
            particles.append(np.array(track))
            
            # Update tqdm progress bar every 1% progress
            completed = l - data.shape[0]
            if completed / l >= 0.01 * u:
                u += 1
                pbar.update(completed - pbar.n)  
                pbar.set_postfix_str(f"{round(100 * completed / l)} %")

        

        particles = [p for p in particles if p.size > 0]
    print("Particles Successfuly Tracked...")
    
    return particles


def tagParticles(filtered_particles, xlsx_save_path: str, tif_path: str, tagged_tif_path: str) -> None:
    """
    Tags particles in a multi-frame .tif image using data from an Excel file, and saves
    the output as a new .tif file with particle indices overlaid on each frame.

    Parameters
    ----------
    data_path : str
        Path to the Excel file (.xlsx) containing particle tracking data. Each sheet 
        should correspond to a particle with at least columns: [area, x, y, frame].
    tif_path : str
        Path to the input binary .tif file (multi-frame image).
    tagged_tif_path : str
        Path to save the output .tif file with particles tagged.

    Notes
    -----
    - Each frame of the .tif is tagged by overlaying particle indices at their (x, y) 
      positions, centered using `anchor="mm"`.
    - A green pixel is also placed at each (x, y) location as a marker.
    - The progress of tagging is displayed using a tqdm progress bar.
    - The output .tif will contain all tagged frames as a multi-frame .tif file.
    """
    particle_data = []
    _, name = os.path.split(xlsx_save_path)

    try:
        with pd.ExcelWriter(xlsx_save_path, engine="openpyxl") as writer:
            for i, particle in enumerate(tqdm(filtered_particles, desc="Saving Particles", unit="particle")):
                df = pd.DataFrame(particle, columns=[
                    "Area", "Centroid_X", "Centroid_Y", "Frame",
                    "Eccentricity", "BBox_X", "BBox_Y", "BBox_W", "BBox_H", 
                    "Major_Axis", "Minor_Axis", "Orientation", "Domain_Color", "Holes"
                ])
                df.insert(0, "Index", i + 1)  # Add Index column (starts at 1)
                particle_data.append(df)

                sheet_name = f"Particle_{i + 1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved {len(filtered_particles)} particles to {name}")

    except IndexError:
        print("Failed to Track... No Domains found. Consider revising settings.")
        return


    image = Image.open(tif_path)
    n_frames = [frame.copy().convert('RGB') for frame in ImageSequence.Iterator(image)]
    font = ImageFont.load_default()

    data_concat = pd.concat(particle_data).values
    data_sorted = data_concat[data_concat[:, 4].argsort()]

    output_frames = []
    for n, img in enumerate(tqdm(n_frames, desc="Tagging Particles")):
        draw = ImageDraw.Draw(img)
        frame_indices = np.where(data_sorted[:, 4] == n)[0]

        for idx in frame_indices:
            particle_id = int(data_sorted[idx, 0])  # "Index"
            x = int(round(data_sorted[idx, 2]))      # Centroid_X
            y = int(round(data_sorted[idx, 3]))      # Centroid_Y

            if y < 0 or y >= img.height or x < 0 or x >= img.width:
                continue

            draw.text((x, y), str(particle_id), fill=(255, 0, 0), font=font, anchor="mm")
            img.putpixel((x, y), (0, 255, 0))  # Mark with green pixel

        output_frames.append(img)

    output_frames[0].save(
        tagged_tif_path, save_all=True, append_images=output_frames[1:]
    )

    print(f"Saved tagged image to: {tagged_tif_path}")
    
    return tagged_tif_path, xlsx_save_path
    
    

def main():
    print("\n\n\nRunning trackAndTagComplete.py. Please respond to all GUI prompts...\n\n\n")
    
    _ = easygui.buttonbox(
        msg="trackAndTagComplete.py\n\n   This program tracks individual domains from a binary multipage .tif file according to input parameters. It also tags each domain with an index.\nOutputs:\n - xxx_TRACKED.xlsx: Tracked domain data file.\n - xxx_TAGGED.tif: .tif file with domain index printed.\n\n\nPlease respond to all the following prompts for optimal results.",
        title="Settings",
        choices=["Continue"]
    )
    
    #  ------------------------------ Settings ---------------------------------
    # The settings can be changed here or when running the program

    # These settings affect the data, not the tracking
    min_area = 1
    max_area = 100000
    min_frames = 1
    max_eccentricity = 1
    max_area_variation = 1 # Decimal percentage only (0-whatever)

    # These settings below will affect the tracking. Generally you want these
    # values flexible for the first track but tighten them during the
    # iteration.
    threshold_factor = 1.0 
    max_areachange = 1 # Decimal percentage only (0-1)
    max_movement = 100
    max_frameskip = 0


    # These are for the iteration. You want to keep the allowed movement low
    # because the program already is guessing where the particle should be.
    max_areachange2 = 1 # Percentages only
    max_movement2 = 50
    max_frameskip2 = 0
    # -------------------------------------------------------------------------
    
    
    s1 = [str(min_area), str(max_area), str(min_frames), str(max_area_variation), str(max_eccentricity), str(threshold_factor)]
    s2 = [str(max_areachange), str(max_movement), str(max_frameskip)]
    s3 = [str(max_areachange2), str(max_movement2), str(max_frameskip2)]

    (file_path, 
    binary, domain_color, border, iterative_tracking,
    max_eccentricity, min_area, max_area, min_frames, max_area_variation, 
    max_areachange, max_movement, max_frameskip,
    max_areachange2, max_movement2, max_frameskip2,
    adaptive_thresh, threshold_factor, slices, xlsx_save_path, bin_tif_path, tagged_tif_path) = getOptions(s1,s2,s3)
    
    
    data, tif_path = preProcess(file_path, bin_tif_path, binary, domain_color, border, max_eccentricity, min_area, adaptive_thresh, threshold_factor, slices)
    filtered_data = removeParticles(data, min_area, max_area, max_eccentricity)
    

    filtered_data_copy = filtered_data.copy()
    particles = trackParticles(filtered_data, max_areachange, max_movement, max_frameskip, None, None)
     
    if iterative_tracking:
        n_frames = int(np.max(data[:,4]))
        dx, dy = average(particles, n_frames)
        particles_copy = trackParticles(filtered_data_copy, max_areachange2, max_movement2, max_frameskip2, dx, dy)
        filtered_particles = filterParticles(particles_copy, max_area_variation, min_frames)
    else:
        filtered_particles = filterParticles(particles, max_area_variation, min_frames)
    
    
    tagged_tif_path, xlsx_save_path = tagParticles(filtered_particles, xlsx_save_path, tif_path, tagged_tif_path)
    
    return tif_path, tagged_tif_path, xlsx_save_path
    
    
if __name__ == "__main__":
    main()
    

