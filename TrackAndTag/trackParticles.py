import easygui, os, importlib, tifffile, getOptions, preProcess,  averageDisplacement
import numpy as np
import pandas as pd
import scipy.io as sio
from openpyxl import Workbook
from PIL import Image, ImageSequence
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from tqdm import tqdm

''' [trackParticles.py] Last Updated: 5/30/2025 by Myles Koppelman '''


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
    
    for particle in tqdm(particles, desc="Removing Unwanted Particles..."):
        areas = particle[:, 0]
        mean_area = np.mean(areas)
        
        if mean_area == 0:
            continue  # Avoid divide-by-zero

        area_flux = np.abs(1 - (areas / mean_area))
        particle_filtered = particle[area_flux <= max_area_variation]

        if particle_filtered.shape[0] > min_frames:
            filtered_particles.append(particle_filtered)

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
        try:
            max_frame = int(np.max(data[:, 3]))
            xm = np.zeros(max_frame + 1)
            ym = np.zeros(max_frame + 1)
        except ValueError:
            xm = 1
            ym = 1

    n = -1  # track index
    u = 1   # for progress tracking
    

    with tqdm(total=l, desc="Tracking...") as pbar:
        while data.shape[0] > 0:
            n += 1
            # Start new track with first detection
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






def main():
    #  ------------------------------ Settings ---------------------------------
    # The settings can be changed here or when running the program

    # These settings affect the data, not the tracking
    min_area = 200
    max_area = 2000000
    min_frames = 1
    max_eccentricity = 1.0
    max_area_variation = 1.0 # Decimal percentage only (0-whatever)

    # These settings below will affect the tracking. Generally you want these
    # values flexible for the first track but tighten them during the
    # iteration.
    threshold_factor = 1.0 
    max_areachange = 0.8 # Decimal percentage only (0-1)
    max_movement = 100
    max_frameskip = 0


    # These are for the iteration. You want to keep the allowed movement low
    # because the program already is guessing where the particle should be.
    max_areachange2 = 0.8 # Percentages only
    max_movement2 = 50
    max_frameskip2 = 0
    # -------------------------------------------------------------------------
    
    
    s1 = [str(min_area), str(max_area), str(min_frames), str(max_area_variation), str(max_eccentricity), str(threshold_factor)]
    s2 = [str(max_areachange), str(max_movement), str(max_frameskip)]
    s3 = [str(max_areachange2), str(max_movement2), str(max_frameskip2)]

    (file_path, 
    binary, domain_color, border,
    max_eccentricity, min_area, max_area, min_frames, max_area_variation, 
    max_areachange, max_movement, max_frameskip,
    max_areachange2, max_movement2, max_frameskip2,
    write_bin,
    adaptive_thresh, threshold_factor, slices, 
    bin_save_path, tracked_save_path) = getOptions.getOptions(s1,s2,s3)
    
    
    data = preProcess.preProcess(file_path, binary, domain_color, border, max_eccentricity, min_area, write_bin, adaptive_thresh, threshold_factor, slices, bin_save_path)
    # filtered_data = removeParticles(data, min_area, max_area, max_eccentricity)
    # filtered_data_copy = filtered_data.copy()
    filtered_data = data
    particles = trackParticles(filtered_data, max_areachange, max_movement, max_frameskip, None, None)
    
    # n_frames = int(np.max(data[:,4]))
    
    # dx, dy = averageDisplacement.average(particles, n_frames)
    # particles = trackParticles(filtered_data_copy, max_areachange2, max_movement2, max_frameskip2, dx, dy)
    # filtered_particles = filterParticles(particles, max_area_variation, min_frames)
    filtered_particles = particles
    
    
    _, name = os.path.split(tracked_save_path)
 
    try:
        with pd.ExcelWriter(tracked_save_path, engine="openpyxl") as writer:
            for i, particle in enumerate(tqdm(filtered_particles, desc="Saving Particles", unit="particle")):
                df = pd.DataFrame(particle, columns=[
                    "Area", "Centroid_X", "Centroid_Y", "Frame",
                    "Eccentricity", "BBox_X", "BBox_Y", "BBox_W", "BBox_H", "Major_Axis", "Minor_Axis", "Orientation", "Domain_Color"
                ])
                sheet_name = f"Particle_{i+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\nSaved {len(filtered_particles)} particles to {name}")
    except IndexError:
        print("Failed to Track... No Domains found. Consider revising setting.")
    
    
    
    

if __name__ == "__main__":
    main()
    

