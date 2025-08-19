import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image, ImageSequence
from tifffile import imwrite
import os
from utils import *

''' [threshold.py] Last Updated: 7/11/2025 by Myles Koppelman'''


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


def threshold(tif_path, tif_save_path):
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
    thresholdGui(frames, tif_save_path)
    return tif_path, tif_save_path
    

def main():
    tif_path = getTif("Select File to Threshold")
    tif_save_path = saveTif("Select path to save Thresholded File", "_BIN", tif_path)
    threshold(tif_path, tif_save_path)


if __name__ == "__main__":
    main()

