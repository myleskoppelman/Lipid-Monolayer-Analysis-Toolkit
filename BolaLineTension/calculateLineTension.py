import easygui, os, bolaLineTension, isolateDomains, filterData, trackAndTagComplete

''' [calculateLineTension.py] Last Updated: 8/8/2025 by Myles Koppelman '''

def main():
    """
    Master script for calculating the line tension of a bola domain structure from a raw .tif video stack.

    This function acts as a pipeline integrating multiple modules:
    - trackAndTagComplete.py: Tracks and tags domain features.
    - filterData.py: Filters data to remove undesired frames/domains.
    - isolateDomains.py: Isolates the bola from other domains.
    - arcLength.py: Calculates the arclength of the bola strip.
    - fitEllipse.py: Calculates the major/minor axes of the bola head.
    - lineTension.py: Calculates the final line tension.
    - drawEllipses.py: Visualizes the final overlay of data.

    The program interacts with the user via EasyGUI dialogs to ensure proper pre-processing.
    It saves intermediate and final results in appropriate formats (TIFF stacks, Excel files), and visualizes these
    using PIL image previews.

    Notes:
    -----
    - This script is intended for well-prepared data. If results are unsatisfactory, rerun each file individually.
    - Ensures consistent tracking of the 'head' and 'base' domains to obtain meaningful line tension values.
    - Users should validate domain tracking manually at multiple stages.
    - For more detailed instruction, see README.md

    Contact:
    --------
    Myles Koppelman (myleskoppelman@icloud.com) (koppe116@umn.edu)
    Project Repository: https://github.com/myleskoppelman/LineTension

    Example Usage:
    --------------
    python calculateLineTension.py
    
    """
    _ = easygui.buttonbox(
        msg="[calculateLineTension.py] Last Updated: 6/6/25 by Myles Koppelman\n\nInstructions: This is the master implementation of lineTension.py. It includes all features to calculate the line tension of a bola from a raw .tif file. This program depends on:    \n  - trackAndTag.py --> Used to track and tag all domains  \n  - filterData.py --> Used to remove undesired domains and frames \n  - isolateDomains.py --> Used to remove all other domains from file other than desired bola  \n  - arcLength.py --> Calculate length of bola strip   \n  - fitEllipse.py --> Calculate radius of bola head  \n  - lineTension.py --> Caclulate value of line tension \n  - drawEllipses.py --> Visualize calculation of line tension \n\n!!!NOTES: If you find this file failing to produce results, I recommend running each file individually as the preprocessing is very important for good results. For example, you must make sure to continuously track the 'head' and 'base' of the bola, or else the calculation will certainly fail. The tracking can take a lot of trial and error to obtain the correct parameters. When in doubt, run each file individually. This file is meant for quickly calculating the line tension for files containing very good data!!!\n\nFor more information, contact 'myleskoppelman@icloud.com' or check the README.md found at:\nhttps://github.com/myleskoppelman/LineTension",
        title="Settings",
        choices=["Continue"]
        )
    

    # ---------------------------------------------------------------------------------------------------------
    # Track and Tag. This part preprocesses raw .tif data and identifies all valid domains according to hyperparameters
    # ---------------------------------------------------------------------------------------------------------
    tif_path, tagged_save_path, idx_save_path  = trackAndTagComplete.main()
    os.system(f"open '{tagged_save_path}'")
    
    
    # ---------------------------------------------------------------------------------------------------------
    # Filter #1: This filter should be done to get rid of excess domains other than the head and base of the bola
    # ---------------------------------------------------------------------------------------------------------
    filtr = easygui.buttonbox(
        msg="Does the data need to be filtered?\nIf your video contains more domains than only the 'head' and 'base' domain, select yes\nIf there are any frames that don't contain both the 'head' and 'base domain(other than when they merge), select yes",
        title="Settings",
        choices=["Yes", "No"]
    )
    filter = False if filtr == "No" else True
    
    if filter:
        filtered_data_path, filtered_tif_path = filterData.getSaveFiles(idx_save_path, tif_path)
        filtered_data_path, filtered_tif_path = filterData.filterData(idx_save_path, tif_path, filtered_data_path, filtered_tif_path)
    else:
        filtered_data_path = idx_save_path
        filtered_tif_path = tif_path
    
    
    
    
    # ---------------------------------------------------------------------------------------------------------
    # Isolate Domains. This should be done to remove all other domains from the file so only head and base remain.
    # This is because to calculate the arc length, only the head and base can be present, see README.md for more info
    # ---------------------------------------------------------------------------------------------------------
    isolate = easygui.buttonbox(
        msg="Do the bola domains need to isolated?\nIf there are other domains in the video than the 'head' and 'base' domain, select yes",
        title="Settings",
        choices=["Yes", "No"]
    )
    iso = False if isolate == "No" else True

    if iso:
        iso_save_path = isolateDomains.drawDomains(filtered_data_path, filtered_tif_path)
        
        os.system(f"open '{iso_save_path}'")
        
        filtr = easygui.buttonbox(
            msg="Does the data need to be filtered again?\n\n!!!NOTE: The 'head' and 'base' domain must track continuously throught the video. If they do not, quit the program and retrack with different settings. Try adjusting 'frameskip' and 'max area variation' parameters if domains do not track continuously!!!\n\nIf your video contains more domains than only the 'head' and 'base' domain, select yes.\nIf there are any frames that don't contain both the 'head' and 'base domain(other than when they merge), select yes.",
            title="Settings",
            choices=["Yes", "No"]
        )
        filter = False if filtr == "No" else True
    
        # ---------------------------------------------------------------------------------------------------------
        # Filter #2: This filter should be used to get rid of any frames in which the head and base are not both present.
        # Again, see README.md for more information on why this is necessary
        # ---------------------------------------------------------------------------------------------------------
        if filter:
            final_data_path, final_tif_path = filterData.getSaveFiles(filtered_data_path, iso_save_path)
            final_data_path, final_tif_path = filterData.filterData(filtered_data_path, iso_save_path, final_data_path, final_tif_path)
        else:
            final_data_path = filtered_data_path
            final_tif_path = iso_save_path
    
    else: 
        final_data_path = filtered_data_path
        final_tif_path = filtered_tif_path
    
    

    # ---------------------------------------------------------------------------------------------------------
    # Calculate Line Tension: Finally, the arc length and head radius is calculated to calculate the arclength.
    # ---------------------------------------------------------------------------------------------------------
    lt_save_path, lt_data_path = bolaLineTension.getSaveFiles(final_tif_path)
    bolaLineTension.lineTension(final_data_path, final_tif_path, lt_save_path, lt_data_path)
    
    os.system(f"open '{lt_save_path}'")
     

if __name__ == "__main__":
    main()