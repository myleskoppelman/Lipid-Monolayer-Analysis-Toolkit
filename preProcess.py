from getOptions import *
from tqdm import tqdm
import tifffile
from PIL import ImageSequence
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, remove_small_objects
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

''' [preProcessFile.py] Last Updated: 5/30/2025 by Myles Koppelman '''


def preProcess(file_path, is_bin, domain_color, border, max_eccentricity, min_area, write_bin, adaptive_thresh, threshold_factor, slices, bin_save_path):
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

    img = Image.open(file_path)
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
        # if not is_bin:
        if adaptive_thresh == 0:        # Normal Threshold
            t = threshold_otsu(im) * threshold_factor 
            binary = im > (t * 0.8)
        else:                           # Adaptive Threshold
            h, w = im.shape
            binary = np.zeros_like(im, dtype=bool)
            block_h = h // horz_slices
            block_w = w // vert_slices
            for i in range(horz_slices):
                for j in range(vert_slices):
                    y0, y1 = i * block_h, (i + 1) * block_h if i < horz_slices - 1 else h
                    x0, x1 = j * block_w, (j + 1) * block_w if j < vert_slices - 1 else w

                    block = im[y0:y1, x0:x1]
                    if block.size == 0:
                        continue
                    t_block = threshold_otsu(block) * threshold_factor
                    binary[y0:y1, x0:x1] = block > t_block

        if domain_color == 0:
            binary = ~binary
            

        binary = binary_opening(binary)
        # ----------------------- Remove these for unedited binary image ---------------------------
        binary = remove_small_objects(binary, min_area) 
        binary = binary_fill_holes(binary)
        if not border:
            binary = clear_border(binary)
        # ------------------------------------------------------------------------------------------
        if write_bin:
            processed_frames.append(img_as_ubyte(~binary))       
        # else:
        #     binary = ~im.astype(bool)


    # ---------------------------------------------------------------------------------------------------------
    # IDENTIFY PARTICLES
    # ---------------------------------------------------------------------------------------------------------
        labeled, num = label(binary, return_num=True, connectivity=2)

        if num > 0:  # [Area, CentroidX, CentroidY, Frame#, Eccentricity, BoundingBoxX, BoundingBoxY, BoundingBoxW, BoundingBoxH]
            props = regionprops(labeled)
            m = np.zeros((len(props), 12))

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

            eccentricities.extend(m[:, 4])
            output_data.extend(m.tolist())

        else:
            print(f"No Domains Found on Frame {n}. Domain Color May Have Reversed.")
            ne += 1


    eccentricities = np.array(eccentricities)
    odata = np.array(output_data)
    if eccentricities.size > 0:
        row = np.where(eccentricities > max_eccentricity)[0]
        odata = np.delete(odata, row, axis=0)


    # Save binary output images
    if write_bin and processed_frames:
        tifffile.imwrite(bin_save_path, processed_frames, dtype=np.uint8)


    print("Threshold and Analysis Complete...")
    
    return odata
    
    
    