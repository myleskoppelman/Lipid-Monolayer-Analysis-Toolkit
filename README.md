# LineTension

Recommendations:

Currently, I would recommend running this program with a .tif file that has already be thresholded into a binary file. The threshold implementation within preProcess.py works, but it doesn't allow you to modfiy any parameters, so you cannot control what the threshold will do. This program is currently very picky, and requires good data with easily identifiable, continuous domains of the 'head' and 'base' of the bola, or it will not work. 

At the moment the program has only been verified to work if you are selecting black domains on a white background.

It is of utmost importance that the 'head' and 'base' of the domain track continuosly, all the way through the movie, until they merge at the end. If your 'head' or 'base' disapears prematurely at any point, you will have to reprocess your data by adjusting the hyperparameters, until they are present. The program is picky, so if you have bad data, it just might not work at all.


# Methodology
Based off “Molecular determinants for the line tension of coexisting liquid phases in monolayers. Andrea Alejandra Bischof, Natalia Wilke”

Fλ = Fη = f * η * a * dL/dt

- f: “a dimensionless coefficient that depends on the rheological properties of the monolayer and the subphase”

- η: viscosity of the subphase

- a: radius of the head of the bola

- dL/dt: change in length of the strip of the bola



# Procedure for calculateLineTension.py
1.  Find suitable domain and apply threshold such that only the ‘head’ and ‘base’ of the domain appear in threshold. Ensure the isolated domains are fully separated from anything else in the data.
  - Once you have a good input file, run calculateLineTension.py



2. 	Firstly, calculateLineTension.py runs your 'exampleName.tif' file through the file trackAndTag.py:

trackAndTag.py has the following dependancies:
- getOptions.py --> Prompts user for tracking hyperparameters.
- preProcess.py --> Applys a threshold to the .tif file (if its not already binary) and identifies regions using scikit-image 'regionprops'.
- averageDisplacement.py --> Calculates the movement of regions based on previous data, used for iterative tracking.

By running trackAndTag.py, you will obtain two files:
- ‘exampleName_DARK_TRACKED.xlsx’ or ‘exampleName_LIGHT_TRACKED.xlsx':
  - This file contains RAW tracking data of every 'region' in the image according to "domain color" choice. The 'regions' are found by the file preProcess.py, while trackParticles() identifies which 'region' corresponds to the same domain, and seperates each domain into its own page of an .xlsx with all the relevant data about the domain. This file is then overwritten by the function tagParticles(), which adds an individual index to each particle.
- ‘exampleName_TAGGED.tif’:
  -  This file is the original .tif file with the index of each domain superimposed on top so the user can identify visually which domains correspond to the 'head' and 'base' of the bola. This file is written to within the function tagParticles().
 





3. 	Then the file ‘exampleName_TAGGED.tif’ will pop up, look through it and take note of:

- Which two domain numbers are the ‘head’ and ‘base’ (if you obtained more than two domains when running trackAndTag.py). You will need to keep track of these domains so you can filter out any other domains.
- Any frames that don’t contain the ‘head’ and ‘base’ domains. Excluding when the head and base merge at the end, if any frames near the beginning do not contain the domains, you will want to take note of the frame number so you can filter it out.






4. Next, a prompt will pop up asking if the data needs filtering. You will want to click yes if:

- There are more domains present in the images than just the 'head' and 'base' domain. You must filter out any excess domains out of the .xlsx file.
  - If you click yes, enter the index of the of the 'head' and 'base' domains, you can leave the 'frames' box empty for now. The data in  ‘exampleName_DARK_TRACKED.xlsx’ and 'exampleName.tif' will be run through the file filterData.py, which will remove all domains not specified above from the .xlsx file. The files will be saved to two new files ending in '_FLTR.xlsx' and '_FLTR.tif'



5. Next, a prompt will apear asking if the domains need to be isolated. You will want to click yes if:

- There are any black regions other than the 'head' and 'base' in your original 'exampleName.tif' file.
  - If so, click yes, and select a path to save your newly isolated data. The file 'exampleName.tif' will then be run through isolateDomains.py, and saved by default with the ending  '_ISO.tif'.
 
  
The purpose of this is to remove any other domains from the images that are not the ‘head’ or the ‘base’. The reason is because the program to calculate the arc length fits a line through all the black pixels in the images, so any other domains remaining in the image will affect the results. If you have other domains in the image, it will try to fit it through those aswell, so the value of the arc length won't equal the distance between the 'head' and 'base' of the bola. This is why it’s important to filter out any other domains other than the ‘head’ and ‘base’ beforehand, or else they won’t be removed by running isolateDomains.py. isolateDomains.py works by reading all domains from the .xlsx data file, and then drawing them onto a blank canvas by drawing any black pixels within the bounding box obtained from scikit-image 'regionprops'. Thus if other domains remain, they will be drawn too and ruin the calculation of the arc length.


6. 	Now ‘exampleName_ISO.tif’ will appear, verify once again that:

- Only the desired domains are in the image.
- There are no frames which are missing the domains.
  - If there are any frames in which only the 'head, or only the 'base' of the domain appear, filter those out. Normally they would be at the start if you have good data. In the worse case scenario where some of them disapear mid movie, I would just reprocess until you are tracking them all the way through.
 
If there are any frames at the start missing the domains, but the rest of the video is continuous, take note of the index of those frames (0-based) and click yes on the prompt to filter again. Make sure you input the index of the head and base, and input the index of the frames that must be removed.




8. 	Now your files will be run through lineTension.py

The program will output a .pdf file containing graphs measuring the arc length, and will also provide graphs detailing the rate of change of the arc length, ‘head’ radius, and line tension over time. All of this data will be saved to 'exampleName_ISO_LT_DATA.xlsx', aswell as a .tif file showing the ellipse and arc length calculated overlaid onto the original .tif file.


# Known Errors That May Arrise:

The program requires a lot of preprocessing and very good data to run correctly at the moment. A lot of errors can result by attempting to plot the data at the end, if the length of the radius array (a) or line tension array(line_tension) are not equal to the array containing the number of frames (x).
- If you have any errors that deal with the function arcLength() in arcLength.py, ensure that the # of frames matches in the .xlsx and .tif files. When in doubt just try to reprocess.
- LET ME KNOW of any other errors that occur (koppe116@umn.edu, myleskoppelman@icloud.com)


# Technical Details: 
1. the radius of the head(a) is determined by using the major and minor axis of the encompassing ellipse of the domain using scikit-image “regionprops” function.
trackParticles.py works by obtaining all ‘regions’ in each frame using ‘regionprops’

‘regionprops’ has many statistics such as centroid, bounding boxes, major and minor axis, and orientation.
You can obtain a visual representation of the elliptical fits using	drawEllipses.py 

Improvement should be made on this feature to obtain a more accurate value of the radius(a). Sometimes the function	tracks the major and minor axis of the 'head' inaccurately due to pixels from the 'strip' of the bola in the frame. For more accurate results, the program should be updated to track only the part we consider the 'head'


2. The rate of change of the strip (dL/dt) is obtained by calculating the arc length between the domains, fitting the data, and taking the derivative of the fit.

The actual arc length is determined using numpy ‘polyfit’ function, fitting on the rows and columns containing black pixels (the domains)


Due to the lack of precision calculating the radius of the head (a), the line tension over time varies, which is not physically accurate. More work will be done to refine the methodology of calculating the line tension and improving the value. 
On the positive side, the results of the line tension are in pico Newtons, which is around what is to be expected for the value of the line tension in these systems. 



# Procedure to run lineTension.py Manually

If you are having issues running calculateLineTension.py, try running each file individually which will help you locate your errors.

1. trackAndTag.py --> Run trackAndTag.py using you binary .tif file
2. Look through the output '_TAGGED.tif' file to ensure the head and base domain appear continuously throughout the video. See YouTube tutorial for more info. If the domains don't track all the way through, either your data will be inaccurate or the program will fail. Try reprocess if they do not track
3. filterData.py --> If more domains are tracked than the 'head' and 'base' domain, take not of the index of the 'head' and 'base' and run the 'TRACKED.xlsx' and original .tif file through filterData.py. Enter the numbers of the desired domains, leave the remove frames box empty for now.
4. isolateDomains.py -->  If there are any domains in your .tif file other than the base and head, run the newly saved '_FLTR.xlsx' and '_FLTR.tif' file through isolateDomains.py
5. Look through the newly saved '_ISO.tif' file and verify only the 'head' and 'base' domains remain in the video. Also ensure both domains appear continuously throughout the whole video.
6. filterData.py --> If there are any frames at the start or end of the video which don't contain both the 'head' and 'base' domain, run the '_FLTR.xlsx' and '_ISO.tif' files through filterData.py again, and take note of which frame # don't have both domains. Enter those frames into the box (aswell as the index of the 'head' and 'base') and click ok. NOTE: if there are frames missing the head or base in the middle of the video, you will need to reprocess.
7. Look through the newly saved filter file and verify now that only the 'head' and 'base' domain remain continuously in the video.
8. lineTension.py --> Now run the final filtered .tif file and .xlxs file through lineTension.py

