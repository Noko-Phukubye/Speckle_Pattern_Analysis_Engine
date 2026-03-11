https://www.markdownguide.org/cheat-sheet/
# Research Log - 27 April 2025 - map_coordinates interpolation order

### Session Goals:
- Deform images using different map_coordinates interpolation settings. The aim was to observe their effect
on error results. 

### What Was Tried:
- Implemented RBF Interpolation on displacement data from FEM model and deformed images were generated using different
'map_coordinates' interpolators.

- Code used:
  ```python
    new_x = grid_x - dx
    new_y = grid_y - dy
    new_coordinates = np.array([new_y,new_x])

    # Remap the image using the TPS-based transformation
    image_interpolation_order = 3
    print("Remapping image using RBF-based transformation and map_coordinates...")
    remapped_image = np.zeros_like(image)
    if len(image.shape) == 2:  # Greyscale image 
            remapped_image = map_coordinates(image, new_coordinates, order=image_interpolation_order, mode='constant')
- order was set to 0, 1, 2, 3

### Outcome:
- Only order 0 (nearest neighbor interpolation) produced significantly different results.
- Nearest neighbor is not convolution-based but it is behaving weirdly at the same regions as the convolution-based
interpolation algorithms.
- The other modes produced the same error graphs


---
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
# Research Log - 28 April 2025 - Code fixing

### Session Goals:
- Fixed various bugs and issues
- speckle.py had an issue where some speckles were defined beyond the boundary
- Sub-pixel analysis code was loading files incorrectly

### What Was Tried:
- Implemented RBF Interpolation on displacement data from FEM model.
- Added a boundary checking section to speckle_image in speckle.py. This skips speckles that are out of bounds.
The following amendment was made to line 26 of the speckle.py source file:

    ```python
        if D < 3:
            D = 3       # Not part of the original src
        #--------------------------------------------
        # if D < 3:
            # raise Exception('Set higher speckle diameter (D >= 3 px)!')
    ```
    

- The following amendment was made to line 101 of the speckle.py file:

    ```python
        if (y - dy >= 0) and (y + dy <= h) and (x - dx >= 0) and (x + dx <= w):
            sl = np.s_[y-dy:y+dy, x-dx:x+dx]
            im[sl] -= s
        else:
            # Skip if the speckle would go outside the image
            continue
    ```
    
- Added an extra line in the subpixel analysis that filters binary files according to u_imp. This ensured that
only the files corresponding to the displacement under study would be loaded. Issue was resolved.

    ```python
        sundic_binary_files = sorted(
            [f for f in os.listdir(translation_bin_folder)
            if f.split('_')[1] == f'T{u}'in f and f.endswith('results.sdic')],
            key=lambda x: int(x.split('_')[0])
        )
    ```
    

### Outcome:
Sub-pixel translation analysis code had a few issues:
- binary files were not filtered for u_imposed
- Even after the lines was added, the value needed to be rounded due to floating point issues such as 0.3 -> 0.300000000000000000004
Speckle.py was altered so as to never produce speckles below the necessary threshold. Added a hard lower boundary for the speckle size but
I am not sure what the implications thereof are. I dont see an issue thus far because the images produced have sufficient variation.

### To consider next time:
- How do variations in parameters translate to variations in error?
- 
---
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
# Research Log - 28 April 2025 - different image remappings with different interpolation settings

### Session Goals
- Looking at different image deformation methods
- Problematic images will be identified and served as the focus for the experiment. 5 images with minimal artifacts,
5 with prominent artifacts and 1 DIC challenge image. 
- The images will also be inverted and binarised. 
- Study map_coordinates and cv2.remap
- Vary interpolation and boundary settings to see the effects on DIC procedure
- Understand why the subpixel analysis is returning wacky results

### What Was Tried
- Code used:
    ```python
    new_x = grid_x - dx
    new_y = grid_y - dy
    new_coordinates = np.array([new_y,new_x])

    # Remap the image using the TPS-based transformation
    image_interpolation_order = 3
    print("Remapping image using RBF-based transformation and map_coordinates...")
    remapped_image = np.zeros_like(image)
    if len(image.shape) == 2:  # Greyscale image 
            remapped_image = map_coordinates(image, new_coordinates, order=image_interpolation_order, mode='constant')
    ```        

- And openCV's version:
    ```python
        map_x = (grid_x - dx).astype(np.float32)
        map_y = (grid_y - dy).astype(np.float32)
        #--------------------------------------------------

        # Remap the image using OpenCV
        print("Remapping image using OpenCV...")
        if len(image.shape) == 2:  # Grayscale image
            remapped_image = cv2.remap(image, map_x, map_y, 
                                    interpolation=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT)
    ```

The following settings were applied to 10 images (per setting) during the map_coordinates implementation:
- order: 0, 1, 2, 3

These were the openCV interpolation settings:
- cv2.INTER_NEAREST
- cv2.INTER_LINEAR
- cv2.INTER_CUBIC
- cv2.INTER_AREA
- cv2.INTER_LANCZOS4

In both cases, the borders take a constant value of 0 intensity. These interpolation schemes can be interpreted to be 
convolution-based.

### Outcome
The outcomes were as follows:
- All parameters show inflection points at regions around intervals of 0.5 pixels deformation (0.5, 1.5, 2.5, 3.5 pixels) occuring at 250, 750, 1250, and 1750 . In these regions, the errors show a change from underestimation to overestimation. Probably due to the symmetric nature of the interpolation algorithms. cv2.INTER_LANCZOS showed less abrupt changes in these regions.
- The same results were obtained for inverted images (Update)
- The results for the binarised images (Update)

---
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
# Research Log - 28 April 2025 - muDIC and TexGen
### Session Goals
- To implement muDIC API
- Generate speckle patterns
- Implement TexGen to try and avoid pixel-translation-based interpolation errors
- Run analyses on image batch from previous study (above)
- (Bonus) Check if the DIC algorithm choice has any effect (IC-GN vs IC-LM)
- muDIC was installed and the attempt to obtain positive results continues
- Reading the TexGen source paper and looking to find the API (if it exists)
- Attempt frequency domain shifting (If time allows)
- The following free DIC codes are listed in literature and other resources:

1. muDIC 
2. CMV_DIC
3. Ncorr
4. DICe


### What was tried
- Installed muDIC, attempting to obtain positive results

### Outcome
- Found paper on which TexGen is based
- Paper shows how to generate and manipulate Perlin noise as patterns
- Was able to generate and implement a case for 2 px deformation. First set of pixel interpolations was
avoided successfully
- Found muDIC API. No successful run yet.
---

<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
# Research Log - 01 May 2025 - Inverted vs Normal vs Binarised
### Session Goals
- 450 images were generated
- Inverted (not smoothed) and as generated (not smoothed) -> I want to see if the perform differently after inversion
- Is there a difference between processing images before and after deformation?

### What was tried
- speckles had a 1.5 standard deviation applied to them individually. The size of the kernel was determined by the speckle diameter.
- The DIC gaussian kernel was set to 5 pixels with a standard deviation of 1
- Note to self - inverted 450 in ref_0, normal 450 in ref_1
- Decided that the issue could be metric evaluation so I will now compare their results by 
extracting corresponding metrics as individual vectors and plotting them against one another.
A one-to-one linear relationship with no shift would indicate that the problem lies with the DIC analysis
and not the metric evaluation.
- The effect of subset size (on 20 images)
- The effect of Gaussian filter size (on 20 images)
- The effect of subset size (on 20 images)
- Created difference images using the original and deformed images
- Collapsed the columns and sliced through the images to create two plots to look for 
obvious indicators of aliasing or artifact generation.

### Outcomes
- Both sets of data produced similar errors within similar magnitudes. They ranged from below 0.002 pixels to around 0.18 pixels
- The resulting errors did not behave similarly with regards to the metrics. The normal images showed significant relationships with the 
metrics they were evaluated against (with some very strong Pearson product-moment correlation values). The data from the inverted images
was more scattered and had less concrete relationships.
- Looking at the error slices, both sets of data show inflection points at regions corresponding to (previous_int + 0.5) pixel shift. 
i.e., 0.5, 1.5, 2.5, and 3.5 pixel shifts all occuring at 250, 750, 1250 and 1750 pixels respectively. 
- What is interesting is that the tendency towards over-estimation and underestimation seems to be inverted between the two image sets.
Particularly in the cases of patterns 18, 20, 56, 70, 96 to name a few. For the inverted images the troughs seem to occur in the regions 
the following sets: 0 to 250, 500 to 750, 1000 to 1250, and 1500 to 1750. The opposite is true for corresponding normal images. Here 
the peaks occur in the same specified regions. 
- The above observation was only seen in images that prodoced minimal NANs and had well defined error plots. These same images, however,
tended to have large discrepencies in terms of error magnitudes. In particular, the inverted images tend to have larger error. 
-An example thereof can be seen in the case of pattern 96. The errors tendencies are inverted between the two and the inverted errors 
are about 10 times higher than their normal counter-parts
- Some patterns did exhibit similar error magnitudes betweent the normal and inverted image sets. This was true for pattern 86 (reference image index)


---
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
<!------------------------------------------------------------------------------------------------------------------------------------------ -->
# Research Log - 06 May 2025 - Perlin noise + Optimisation + Nastran
### Session goal
- Deform at 2× resolution, then downsample to see if that does anything
- Optimise according to pattern metrics similar to how I am optimising according to pattern parameters
- The result could be a link to finding a new metric
- Determine what is the issue with the integrated Perlin noise function. The errors look wonky
- Finally implement the muDIC (Astrisk)
- What could limit the reproducability of this porject? I currently suspect that the use of Genesis instead of Nastran should be investigated
. If I can produce the same results through Nastran then I will have two available methods and there will be a lower limitation 
to on the reproducibility. So another goal for this session is to successfully load the Nastran licence and run Nastran using 
the Apex output file that I have
- Generate constant strain files
- Determine the reason as to why the standard deviation vs MIG gives the same graph as RMSE vs MIG (RMSE_d2f = np.sqrt(np.mean((err_on_dic) ** 2)) vs SDE_d2f = np.std(err_on_dic))
Calculate the values manually without using numpy

### What was tried:
I am facing an issue with the Perlin images code. The imposed displacement field is reflected in the DIC error maps. 
The collapsed error curves seem to increase from 0 to + u_imp (or -u_imp when the images are swapped). It looks like 
the error itself shows a u_meas of 0. This could mean that the displacements might not be getting added to the relevant
noise coordinates. This could be due to the applied scale. 


### Outcomes
- The Perlin noise approach definately removes the error associated with the first interpolation step by removing the step entirely. I found
that it definately adds or reduces the amplitude while not affecting the periodicity. Sometimes it just adds little ripples to an existing
periodic profile.
---

<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 13 May 2025 - Characterising the interaction between the two interpolation methods
### Session goals
- Determine the relationship between the pre-DIC and DIC image interpolations. Will amend the slicing data to save numpy files. Will add a section to reload the data. I want to plot the difference betweem the two datas.

- Make read excel a function to use in slice comparison
- study the effects of subset size
- does changing GN-LM allow you to analyse threshold images
- Add use last + 1 column for differentiating between image generation methods. Read that during optimisation
- Look for high dimensional relationships between metrics and errors
- Run inverted Perlin
- Run binarised Perlin (try different Gaussian prefilterings)
- sites: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
https://stackoverflow.com/questions/72958874/plotting-the-1d-power-spectrum-of-an-image

### What was tried
- Methodology for studying the effects of pixel deformation interpolation on DIC results
1. Generate n analytical images (2n image pairs)
2. Deform the n reference image pairs using map_coordinates to create n deformed images while generating excel sheet of parameters with n entries
3. Copy the reference images to create 2n referenc images
4. Move the existing Perlin deformed images to deformed folder to add to the map_coord deformed images
5. There are now 2n reference images and 2n deformed images. n images pairs where deformed analytically and the other half were deformed externally (map_coordinates)
6. Perform DIC analysis on all the images
7. Perform error analysis on all images and save 2n error files (files currently contain column-wise collapsed errors and x_values)
8. Calculate the difference between corresponding error slices. That difference is probably an indication of the pre-DIC interpolation effects
9. Correlate the mean difference per image with image metric value



### Outcomes
- The amount of error that can be attributed to the first interpolation step is definately 
dependent on the pattern characteristics. There is a clear relationship between the additional
error and pattern metrics. This also means that it can be predicted and possibly allow me to 
adjust the RMSE values of various patterns in the excel sheet. It could also explain why I 
errors seem to get worse with higher MIG values for twice interpolated images while the opposite
trend is seen with the perlin images. Its still not to say that they would behave identically even
with the removal of the predicted additional error. This predicted error itself is only based on 
completely different patterns altogether.
---

<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 26 May 2025 - Characterising the interaction between the two interpolation methods (Perlin noise cubic)
### Session goals
- Determine the relationship between the pre-DIC and DIC image interpolations. Created a new script that generates 2N images with 
only N of them being unique
- Tested 300 perlin noise images with a cubic texture function
- Create a surrogate model of error deviation and apply it to the existing data sets
- Assume that the errors are additive. No need to go into great detail with this aspect

### What was tried
- Methodology for studying the effects of pixel deformation interpolation on DIC results
1. Generate n analytical images (2n image pairs)
2. Deform the n reference image pairs using map_coordinates to create n deformed images while generating excel sheet of parameters with n entries
3. Copy the reference images to create 2n referenc images
4. Move the existing Perlin deformed images to deformed folder to add to the map_coord deformed images
5. There are now 2n reference images and 2n deformed images. n images pairs where deformed analytically and the other half were deformed externally (map_coordinates)
6. Perform DIC analysis on all the images
7. Perform error analysis on all images and save 2n error files (files currently contain column-wise collapsed errors and x_values)
8. Calculate the difference between corresponding error slices. That difference is probably an indication of the pre-DIC interpolation effects
9. Correlate the mean difference per image with image metric value

10. Make a surrogate model
11. Load existing data and correct it using surrogate model
12. Save new plots and run optimisation based on corrected values to observe the results.

### Outcomes
- We found that the degree of amplification of the error results by addition of
the first pixel interpolation step depends on the pattern characteristics.
The relationship is demonstrated by plotting the amplitude ratio (combined_amplitude / noise_amplitude) against the pattern assessment metrics. The relationship shows more attenuation corresponding to better metric values. The worse metric values are associated
with amplitude amplification.
---

<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 27 May 2025 - Thesis writing
### Session goals
- Update thesis structure (main and subheadings)
- Finish writing up the sections that are already in the document

### What was tried (This is the preliminary TOC with some descriptions)
1. Introduction
1.1. Background
1.2. Aim and objectives
1.3. Scope of research
1.4. Motivation

2. Literature review
---

<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 27 May 2025 - Optimisation
### Session goals
- Running the entire pipeline without Perlin pairs is faster and more flexible. If I can show that the corrected data is similar to the analytical translation data then I can say that the correction worked.
- I currently have a correction model that has been trained on 300 noiseless Perlin images with a cubic texture. I will now run 75 Perlin images (none, cubic, sinusoid) to see if the model produces accurate results. 
- If it does then I will apply it to the speckles function but the rest of the images will be generated analytically
- Can i generate perlin pairs and then apply the turing function or do i need to generate single images, deform them using grid-based methods and then apply the turing function? There should not be an issue with the first method
 - If I can just add a turing transformation to the already existing image pairsit will save a lot of time. My issue is that
 uncorrelated noise will affect the image correlation. This problem could be alleviated by increasing the repetitions in the turing function.

 - Optimisation strategies:
    1. All metrics used
    2. Strongest 3 (uncorrelated metrics) used
    3. One metric at a time to see the kinds of patterns that are favoured by the metrics

### What was tried 
- NA

### Outcomes
- Still need to run


<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 18 June 2025 - Cover the bases
### Session goals
Entering the final leg of the testing. Some final testing will be done as well as redoing some things I did
in the past.

1. Image generation factors that my influence results
- Different image sizes at same aspect ratio and deformation
- Applying pseudoturing transformation 
* Apply to ref image then deform
* Deform ref and then transform both seperately
*             

2. FE simulation factors that could influence results
- Mesh density
- Deformation size at constant aspect ratio and pixel density

3. Image deform related factors that could influence results
- Interpolation basis function
- Interpolation order

4. DIC settings that could influence results
- Gaussian blur
- ROI (The effect of borders) - how does it relate to interpolation boundary behaviour
- Subset size
- Step size (particularly on checkerboard)
- Shape functions (should be irrelevant due to ultimate error)


Some methods of showing deformation compliance and accuracy
- Use more complex deformations to show accuracy of deformation
- Is inverse consistency applicable here? If so, how?
- What can collapsed difference images tell us other than the difference between zero order and higher order image interpolation?
- How do I ensure that no artifacts are introduced before deformation/DIC ?
- Show displacement field -> How can I evaluate it?
- Show strain field -> can this be used to evaluate smooth field
- The aim is to show that the images are deformed as honestly as possible, 
without unexpected or unintended effects or inaccuracies, before analysis
- Working with greyscale -> Conversion, channel extraction, state of the art, uint8

5. Error analysis methods that my influence results
- Griddata vs RBF
- Interpolation order
- NAN threshold (why 15%)

### What was tried 
- Image sizes [2000x500, 4000x1000, 1000x250] - apply 4mm deformation

### Outcomes
- Still need to run



<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------------------------------------------->
# Research Log - 03 August 2025 - Subpixel study (again)
### Session goals
Trying to understand why I get erratic results when when I perform the subpixel analysis. 

The patterns either look sinusoidal and align with theoretical estimates or they are jagged and
completely miss the mark. The focus is on traditional patterns. 

I thought the issue was the truncated ROI but that is not consistent.
One theory is that the DIC images were resized because the DPI parameter was not 
kept constant at 25.4 but these images performed well before

Fourier shifted images do not result in the proper bias curves

16:05 - I found the current issue, I was passing the images to the translation 
algorithms as float32 datatypes.

```python
    ref_image_path = os.path.join(reference_image_path, ref_file)

    # This caused an issue with the deformed images
    # reference_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    reference_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)  # This works perfectly with Fourier shift
    # reference_image = readImage(ref_image_path)
    
    def image_shift_fourier_scipy(image, u, v):
        """
        Apply subpixel image translation using scipy's fourier_shift.
        
        Parameters:
            image (np.ndarray): 2D grayscale image 
            u (float): Horizontal shift in pixels (x-direction, right is positive) 
            v (float): Vertical shift in pixels (y-direction, down is positive)

        Returns:
            np.ndarray: Shifted image with same shape and dtype as input
        """
        assert image.ndim == 2, "Only 2D grayscale images supported."
        
        shift = [v, u]  
        
        # Take FFT
        image_fft = np.fft.fft2(image)
        
        # Apply fourier shift
        shifted_fft = fourier_shift(image_fft, shift)
        
        # Take inverse FFT and return real part
        shifted_image = np.real(np.fft.ifft2(shifted_fft))
        
        return shifted_image.astype(image.dtype)    

    ```
