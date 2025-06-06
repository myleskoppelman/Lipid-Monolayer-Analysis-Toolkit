import os
import easygui
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile

''' [tagParticles.py] Last Updated: 6/6/2025 by Myles Koppelman '''



def getFiles() -> tuple[str, str, str]:
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
        default=os.path.join(path, f"{nm}_TAGGED.tif"),
        filetypes=["*.tif"]
    )
    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.tif'):
        save_path += '.tif'
    
    return data_path, tif_path, save_path





def tagParticles(data_path: str, tif_path: str, save_path: str) -> None:
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
    save_path : str
        Path to save the output .tif file with particles tagged.

    Notes
    -----
    - Each frame of the .tif is tagged by overlaying particle indices at their (x, y) 
      positions, centered using `anchor="mm"`.
    - A green pixel is also placed at each (x, y) location as a marker.
    - The progress of tagging is displayed using a tqdm progress bar.
    - The output .tif will contain all tagged frames as a multi-frame .tif file.
    """
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    
    path, filename = os.path.split(data_path)
    nm, _ = os.path.splitext(filename)
    new_data_path = os.path.join(path, f"{nm}_IDX.xlsx")


    data = []
    with pd.ExcelWriter(new_data_path, engine='openpyxl') as writer:
        for n, sheet_name in enumerate(xls.sheet_names, start=1):
            df = pd.read_excel(xls, sheet_name=sheet_name)
            index_col = np.full((df.shape[0], 1), n)  # column of n's
            df_with_index = pd.concat([pd.DataFrame(index_col, columns=['Index']), df], axis=1)
            data.append(df_with_index)
            df_with_index.to_excel(writer, sheet_name=sheet_name, index=False)

    image = Image.open(tif_path)
    n_frames = [frame.copy().convert('RGB') for frame in ImageSequence.Iterator(image)] 
    
    font = ImageFont.load_default()

    data_concat = np.vstack(data)     
    data_sorted = data_concat[data_concat[:, 4].argsort()]

    output_frames = []
    for n, img in enumerate(tqdm(n_frames, desc="Tagging Particles")):
        frame_array = np.array(img)
        draw = ImageDraw.Draw(img)

        for idx in np.where(data_sorted[:, 4] == n)[0]:
            particle_id = int(data_sorted[idx, 0])
            x = int(round(data_sorted[idx, 2]))
            y = int(round(data_sorted[idx, 3]))

            if y < 0 or y >= frame_array.shape[0] or x < 0 or x >= frame_array.shape[1]:
                continue
            
            text = str(particle_id)
    

            draw.text((x, y), text, fill=(255,0,0), font=font, anchor="mm")
            img.putpixel((x, y), (0, 255, 0))

        output_frames.append(img)


    output_frames[0].save(
        save_path, save_all=True, append_images=output_frames[1:]
    )
    
    print(f"Saved tagged image to: {save_path}")
    
    


    
def main():
    data_path, tif_path, save_path = getFiles()
    tagParticles(data_path, tif_path, save_path)
    
    
if __name__ == "__main__":
    main()
    


