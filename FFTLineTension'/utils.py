import easygui
import os
import math


def getTif(prompt: str) -> str:
    tif_path = easygui.fileopenbox(
        msg=prompt,
        default="Data#",
        filetypes=["*.tif"]

    )
    if not tif_path:
        raise Exception("No file selected.")
    
    return tif_path


def saveTif(prompt: str, tag: str, old_path: str) -> str:

    path, filename = os.path.split(old_path)
    name, _ = os.path.splitext(filename)
    
    save_path = easygui.filesavebox(
        msg=prompt,
        default=os.path.join(path, f"{name}{tag}.tif"),
        filetypes=["*.tif"]
    )

    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.tif'):
        save_path += '.tif'
        
    return save_path


def getXlsx(prompt: str) -> str:
    xlsx_path = easygui.fileopenbox(
        msg=prompt,
        default="Data#",
        filetypes=["*.xlsx"]
    )
    
    if not xlsx_path:
        raise Exception("No file selected.")
    
    return xlsx_path




def saveXlsx(prompt: str, tag: str, old_path: str) -> str:
    path, filename = os.path.split(old_path)
    name, _ = os.path.splitext(filename)
    
    save_path = easygui.filesavebox(
        msg=prompt,
        default=os.path.join(path, f"{name}{tag}.xlsx"),
        filetypes=["*.xlsx"]
    )
    
    if not save_path:
        raise Exception("No output file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.xlsx'):
        save_path += '.xlsx'
        
    return save_path


def sigfigs(x, sigfigs):
    if x == 0:
        return 0
    return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)


