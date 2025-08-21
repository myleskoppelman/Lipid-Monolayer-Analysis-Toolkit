import easygui
import os
import math

''' [utils.py] Last Updated: 7/11/2025 by Myles Koppelman '''


def getTif(prompt: str) -> str:
    tif_path = easygui.fileopenbox(
        msg=prompt,
        default="Data#",
        filetypes=["*.tif"]

    )
    if not tif_path:
        raise Exception("No input .tif file selected.")
    
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
        raise Exception("No output .tif file selected.")
    
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
        raise Exception("No input .xlsx file selected.")
    
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
        raise Exception("No output .xlsx file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.xlsx'):
        save_path += '.xlsx'
        
    return save_path


def savePdf(prompt: str, tag: str, old_path: str) -> str:
    path, filename = os.path.split(old_path)
    name, _ = os.path.splitext(filename)
    
    save_path = easygui.filesavebox(
        msg=prompt,
        default=os.path.join(path, f"{name}{tag}.pdf"),
        filetypes=["*.pdf"]
    )
    
    if not save_path:
        raise Exception("No output .pdf file selected.")
    
    if save_path is not None and not save_path.lower().endswith('.pdf'):
        save_path += '.pdf'
        
    return save_path


def sigfigs(x, sigfigs):
    if x == 0:
        return 0
    return round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)