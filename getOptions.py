import os
import easygui
import numpy as np
from PIL import Image

''' [getOptions.py] Last Updated: 5/30/2025 Myles Koppelman '''


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
    write_binary_img = False
    binary_save_path = None
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
    # Ask Domain Color
    # ---------------------------------------------------------------------------------------------------------
    border_ = easygui.buttonbox(
        msg="Consider Domains touching the border?",
        title="Settings",
        choices=["Yes", "No"]
    )

    border = False if border_ == "No" else True
    
    
    
    
    # ---------------------------------------------------------------------------------------------------------
    #  Ask for threshold settings and wether to save binary images
    # ---------------------------------------------------------------------------------------------------------
    if not binary:
        adaptive_thresh = easygui.buttonbox(
            msg="Select Threshold",
            title="Settings",
            choices=["Normal", "Adaptive"]
        )
        
        write_images = easygui.buttonbox(
            msg="Write Binary Images?",
            title="Settings",
            choices=["Yes", "No"]
        )
        
        write_binary_img = True if write_images == "Yes" else False


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
        
    if not binary:  
        if at:
            nm = nm + "_ADPT"
        else:
            nm = nm + "_NORM"
        
    
    # ---------------------------------------------------------------------------------------------------------
    #  Ask location to save files
    # ---------------------------------------------------------------------------------------------------------
    if write_binary_img:
        binary_save_path = easygui.filesavebox(
            msg="Save Output File",
            default=os.path.join(p, f"{nm}_BINARY.tif"),
            filetypes=["*.tif"]
        )
        if not binary_save_path:
            raise Exception("No output file selected.")
        

    data_save_path = easygui.filesavebox(
        msg="Save Output File",
        default=os.path.join(p, f"{nm}_TRACKED.xlsx"),
        filetypes=["*.xlsx"]
    )
    if not data_save_path:
        raise Exception("No output file selected.")
    
    if data_save_path is not None and not data_save_path.lower().endswith('.xlsx'):
        data_save_path += '.xlsx'
        
    if binary_save_path is not None and not binary_save_path.lower().endswith('.tif'):
        binary_save_path += '.tif'



    print("Options Complete...")
    return [
        file_path,
        binary,
        dc,
        border,
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
        write_binary_img,
        at,
        threshold_factor,
        (horz_slices, vert_slices),
        binary_save_path,
        data_save_path
    ]

