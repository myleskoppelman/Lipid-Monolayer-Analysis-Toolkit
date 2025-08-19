import os
import easygui
import pandas as pd
import tifffile

''' [filterData.py] Last Updated: 6/6/2025 by Myles Koppelman '''



def getFiles() -> tuple[str, str, str, str]:
    """
    Opens file dialogs for the user to select the input data file, binary .tif file, 
    and the output .tif file path for saving.

    Returns
    -------
    tuple of str
        A tuple containing:
        - data_path: Path to the selected Excel data file (.xlsx).
        - tif_path: Path to the selected binary .tif file.

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



def getSaveFiles(data_path, tif_path) -> tuple[str, str]:
    """
    Opens file dialogs for the user to select the path to save output data files.

    Returns
    -------
    tuple of str
        A tuple containing:
        - save_path: Path to save the output .tif file.
        - data_save_path: Path to save filtered .xlsx file

    Raises
    ------
    Exception
        If the user does not select a file in any of the dialogs.
    """

    p1, filename = os.path.split(data_path)
    name1, _ = os.path.splitext(filename) 
    p2, filename2 = os.path.split(tif_path)
    name2, _ = os.path.splitext(filename2) 
    
    
    data_save_path = easygui.filesavebox(
        msg="Save Output .xlsx File",
        default=os.path.join(p1, f"{name1}_FLTR.xlsx"),
        filetypes=["*.xlsx"]
    )
    if not data_save_path:
        raise Exception("No output file selected.")
    
    if data_save_path is not None and not data_save_path.lower().endswith('.xlsx'):
        data_save_path += '.xlsx'
    
    save_path = easygui.filesavebox(
        msg="Save Output .tif File",
        default=os.path.join(p2, f"{name2}_FLTR.tif"),
        filetypes=["*.tif"]
    )
    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.tif'):
        save_path += '.tif'
        

    
    return data_save_path, save_path




def filterData(data_path, tif_path, filtered_data_path, filtered_tif_path):
    """
    Filters out unwanted particles (by index) and frames (by index) from an Excel file and corresponding TIFF file.

    Parameters:
    ----------
    data_path : str
        Path to the Excel file containing data for all particles (each particle in a separate sheet).
    tif_path : str
        Path to the multi-page TIFF file containing images corresponding to the data.
    filtered_data_path : str
        Path to save the filtered Excel file.
    filtered_tif_path : str
        Path to save the filtered TIFF file.

    Returns:
    -------
    tuple
        A tuple containing the paths to the filtered Excel file and the filtered TIFF file.

    Notes:
    -----
    - The user is prompted via an easygui box to specify:
        - The indices of the particles to keep (e.g., "1,2,3").
        - The indices of frames to remove (e.g., "0,5,10").
    - The function removes the specified frames from:
        - Each selected particle's data in the Excel sheets.
        - The corresponding frames in the TIFF image stack.
    - The filtered data is saved into new files.
    """
    
    
    settings = easygui.multenterbox( # ask for values
        msg="Enter the index of desired particles to keep, and the number of any frames to remove. Separated values by commas (i.e: 1,2,4,7...)",
        title="Settings",
        fields=["Particles to keep: ", "Frames to remove (indexing starts at 0): "],
        values= ["1,2", ""]
    )
    
    try: # parse values 
        particles = [int(s.strip()) for s in settings[0].split(',') if s.strip() != '']
        frames = [int(s.strip()) for s in settings[1].split(',') if s.strip() != '']
    except ValueError:
        raise ValueError("Invalid input. Make sure to enter numbers.")
    
    

    xls = pd.ExcelFile(data_path, engine='openpyxl') # open files
    tiff = tifffile.imread(tif_path)
    

    filtered_excel = {} # filter out pages and particles from .xlsx file
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty or 'Index' not in df.columns:
            print(f"Skipping {sheet_name}: empty or missing 'index' column.")
            continue

        particle_index = df['Index'].iloc[0]
        if particle_index in particles:
            df_filtered = df[~df['Frame'].isin(frames)].reset_index(drop=True)
            filtered_excel[sheet_name] = df_filtered
        else:
            print(f"Removed Particle with index {particle_index} in sheet '{sheet_name}'")
            
            
    # filter out frames from .tif file
    frames_to_keep = [i for i in range(tiff.shape[0]) if i not in frames]
    tiff_filtered = tiff[frames_to_keep, ...]
    

    # save new data to output files
    with pd.ExcelWriter(filtered_data_path) as writer:
        for sheet_name, df in filtered_excel.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    tifffile.imwrite(filtered_tif_path, tiff_filtered)
    
    print(f"Filtered Data saved to {filtered_tif_path}")
    
    return filtered_data_path, filtered_tif_path
    
    
def main():
    data_path, tif_path, = getFiles()
    filtered_data_path, filtered_tif_path = getSaveFiles(data_path, tif_path)
    filterData(data_path, tif_path, filtered_data_path, filtered_tif_path)
    
    
if __name__ == "__main__":
    main()