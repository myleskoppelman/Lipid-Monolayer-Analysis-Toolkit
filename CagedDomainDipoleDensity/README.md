# Caged Domains Analysis Tool (`cagedDomains.py`)

**Author:** Myles Koppelman  
**Last Updated:** 07/11/2025  

---

## Overview

This Python tool automates the detection, visualization, and analysis of **caged lipid domains** in monolayer images. The program is based on Harden M. McConnell's paper: *“Elementary theory of Brownian motion of trapped domains in lipid monolayers”*.  

The tool allows users to:  

- Identify when smaller lipid domains are fully enclosed within larger domains (“caged”).  
- Generate new binary `.tif` images with highlighted caged regions.  
- Create merged `.xlsx` files with centroid distances between inner and outer domains.  
- Calculate the dimensionless η (eta) parameter for each caged domain pair.  
- Compute the dipole density (μ²).  
- Produce a histogram of scaled η values with an exponential fit, saved as a PDF.  


## Workflow Summary

## 1. Inputs Required

- A **binary TIFF stack** containing domain images.  
- An **inner domains Excel file** (`.xlsx`) with particle measurements for smaller domains.  
- An **outer domains Excel file** (`.xlsx`) for larger enclosing domains.  



## 2. Instructions to Run

### 1. Track inner and outer domains using `trackAndTagComplete.py`:
```bash
python trackAndTagComplete.py
```
- You will have to track twice, once selecting either light or dark for your inner domains, and a second time selecting the opposite color for the outer domains.
- For tracking inner(caged) domains, estimate the range of areas of all the domains in your video (e.g. 100-500px), and keep the minimum frames low (e.g. 3-10 frames) because it matters less that the domains are tracked continuously.
- For tracking outer domains, also estimate the range of areas (e.g. 1000-10000px), but keep the minimum frames higher (e.g. 20-30 frames)
  - NOTE: To run caged domains, it doesn’t matter if there are more domains being tracked other than the caged ones, the program identifies which domains are caged and which aren't as long as you have tracked the inner and outer domains. The point of honing in on a range is just to reduce runtime and file size for the tracking data. It also doesn’t matter if the domains are tracked continuously because the calculations are done frame by frame.


### 2. Run the caged domains analysis:
```bash
python cagedDomains.py
```
Follow the on-screen prompts to select:
  - Binary .tif file
  - Inner domains .xlsx file
  - Outer domains .xlsx file

The program will:

-  Detect caged domain pairs.
-  Save a new .tif stack with highlighted domains (*_CAGED.tif).
-  Save centroid distance data in a merged .xlsx file (*_CAGED_DATA.xlsx).
-  Prompt for a location to save the η analysis PDF (*_DATA.pdf).
-  Display and save a histogram of scaled η values with an exponential fit.


## 3. Processes


### a. Caged Domain Detection
Identifies which inner domains are fully contained within any outer domain using an ellipse boundary check.



### b. Image & Data Output
Generates a new .tif stack with matched outer domains highlighted. Saves centroid distance data for each matched pair to an .xlsx file.



### c. η Calculation
Computes the dimensionless η for each inner–outer domain pair:

𝜂 = (𝑟^2 * 𝜌^2) / 𝑅^3

where:

r = radius of inner domain (from area)

R = radius of outer domain (from area)

ρ = distance between centroids

### d. Plots and fits a histogram of scaled η values (η / mean η) to an exponential decay model to extract dipole density (μ²).



## Outputs:
* xxx_CAGED.tif	Binary image stack with highlighted caged domains
* xxx_CAGED_DATA.xlsx	Excel file with merged inner/outer domain data
* xxx_DATA.pdf	Histogram of η/mean(η) with exponential fit and calculated μ²

η and Dipole Density (μ²) Calculation
η is calculated from domain sizes and centroid distances.

Scaled η values are fitted to:

𝑓
(
𝑥
) =
𝐴
𝑒
−
𝑥
f(x)=Ae 
−x
 
Dipole density is calculated using:

𝜇 =
2𝑘𝐵𝑇 / 3pi * mean(η)

​
 
where: 

𝑘𝐵 is Boltzmann's constant and  𝑇 = 298 kelvin

## Dependencies:

* numpy
* pandas
* matplotlib
* tifffile
* skimage
* scipy
* tqdm
* easygui
* Custom: utils.py



## Notes & Limitations:

*Domain tracking .xlsx files must contain: Index, Frame, Centroid_X, Centroid_Y, Area, Orientation.
*Requires manual file selection via pop-up dialogs.
*Binary .tif images must be thresholded before running the analysis.
