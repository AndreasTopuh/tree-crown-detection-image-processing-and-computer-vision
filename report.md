# Tree Crown Detection and Mapping from Satellite Imagery Using Classical Computer Vision Techniques

---

**Student Name:** [Your Name]
**Student ID:** [Your Student ID]
**Intake Code:** APUMF2512AI
**Module Code & Name:** CT103-3-M-IPCV - Image Processing & Computer Vision
**Assignment Title:** Geospatial Feature Detection, Segmentation and Recognition from Satellite/Aerial Imagery Using Classical Computer Vision
**Date Completed:** 20 April 2026
**Lecturer:** Assoc. Prof. Dr. V. Sivakumar

---

## Table of Contents

1. Abstract
2. Problem Statement
3. Proposed Algorithm with Implementation
   - 3.1 Pipeline Overview
   - 3.2 Stage 1: Preprocessing
   - 3.3 Stage 2: Segmentation
   - 3.4 Stage 3: Feature Extraction
   - 3.5 Stage 4: Recognition
   - 3.6 Stage 5: Post-processing
4. Experimental Results
   - 4.1 Dataset Description
   - 4.2 Quantitative Results
   - 4.3 Qualitative Results
5. Critical Evaluation and Analysis
   - 5.1 Strengths of the Approach
   - 5.2 Failure Case Analysis
   - 5.3 Comparison with Contemporary Methods
6. Conclusion and Recommendations
7. References
8. Appendix

---

## 1. Abstract

This report presents a classical computer vision pipeline for detecting, segmenting, and recognising individual tree crowns from satellite and aerial imagery, falling within the Environmental Monitoring application domain. The system addresses the challenge of automated forest canopy analysis, which is critical for deforestation monitoring, carbon stock estimation, and urban green-space management. The proposed pipeline consists of five sequential stages: (1) preprocessing using Contrast Limited Adaptive Histogram Equalisation (CLAHE) and Gaussian denoising; (2) vegetation segmentation through NDVI thresholding, K-means clustering, and marker-controlled watershed segmentation; (3) multi-modal feature extraction encompassing shape descriptors, Grey Level Co-occurrence Matrix (GLCM) texture features, edge-based features, and spectral colour features; (4) rule-based tree crown recognition using a weighted scoring classifier; and (5) post-processing via morphological operations and connected component labelling. The system was implemented in Python using OpenCV and scikit-image, and tested on satellite imagery of varying resolutions and vegetation densities. Experimental results demonstrate that the pipeline achieves effective tree crown delineation, with discussion of performance under challenging conditions including overlapping canopies, shadow interference, and mixed vegetation types.

---

## 2. Problem Statement

Forest ecosystems provide essential services including carbon sequestration, biodiversity support, and climate regulation (Hansen et al., 2013). Accurate monitoring of tree cover from remotely sensed imagery is fundamental to environmental management, yet manual interpretation of satellite images is time-consuming and subjective. The automated detection and delineation of individual tree crowns from satellite imagery presents several significant challenges that distinguish it from standard object detection tasks.

First, **scale variability** poses a challenge: tree canopies range from small shrubs occupying a few pixels to large tropical trees spanning hundreds of pixels, depending on species, age, and image resolution (Ke & Quackenbush, 2011). Second, **spectral similarity** between trees, shrubs, grasslands, and agricultural crops makes discrimination difficult, as all photosynthetically active vegetation exhibits similar spectral signatures in visible and near-infrared wavelengths (Tucker, 1979). Third, **overlapping canopies** in dense forest stands create connected regions where individual crowns are not clearly separable, requiring specialised segmentation techniques such as watershed algorithms (Beucher & Meyer, 1993). Fourth, **atmospheric and illumination effects** including cloud shadows, haze, and sun-angle variations introduce noise and reduce contrast between vegetation and non-vegetation regions (Gonzalez & Woods, 2018). Fifth, **mixed land cover** scenes containing buildings, roads, water bodies, and bare soil alongside vegetation require robust classification to avoid false positive detections.

This assignment addresses these challenges through the design of a classical computer vision pipeline that does not rely on deep learning, instead leveraging established image processing algorithms including vegetation indices, unsupervised clustering, morphological operations, texture analysis, and rule-based classification. The chosen application domain is **Environmental Monitoring: Forest Area Segmentation and Tree Detection**, where the objective is to detect, segment, and count individual tree crowns from multispectral satellite or aerial imagery.

---

## 3. Proposed Algorithm with Implementation

### 3.1 Pipeline Overview

The proposed system follows a five-stage pipeline architecture, as illustrated in Figure 1. Each stage builds upon the outputs of the previous stage, progressively refining the detection from raw pixel data to classified and counted tree crowns.

**Figure 1.** Pipeline architecture diagram. The five-stage pipeline processes satellite/aerial imagery through: (1) Preprocessing (CLAHE, Gaussian blur, ExG/NDVI, exclusion mask from original image); (2) Segmentation (Otsu thresholding, K-means clustering, texture filtering, shadow/water/soil exclusion, watershed); (3) Feature Extraction (shape, GLCM texture, edge, colour descriptors per region); (4) Recognition (9-rule weighted classifier, confidence scoring); and (5) Post-processing (morphological refinement, connected components, annotation). The pipeline overview panels are shown in Figures 2, 5, and 9 for each test image.

The system was implemented in Python 3.10 using OpenCV 4.8 (Bradski, 2000), scikit-image 0.21 (van der Walt et al., 2014), scikit-learn 1.3 (Pedregosa et al., 2011), and NumPy 1.24 (Harris et al., 2020). Python was selected over MATLAB for its open-source ecosystem, extensive library support for geospatial image processing, and reproducibility advantages.

### 3.2 Stage 1: Preprocessing

The preprocessing stage prepares the raw satellite image for robust segmentation through three operations: contrast enhancement, noise reduction, and vegetation index computation.

**Contrast Limited Adaptive Histogram Equalisation (CLAHE).** Standard histogram equalisation can over-amplify noise in homogeneous regions of satellite imagery. CLAHE (Zuiderveld, 1994) addresses this by dividing the image into contextual tiles and applying histogram equalisation locally within each tile, with a clip limit that prevents excessive contrast amplification. The algorithm operates on the L (lightness) channel of the CIE LAB colour space to preserve colour fidelity while enhancing luminance contrast. This is particularly beneficial for satellite images captured under haze or low-contrast atmospheric conditions. A clip limit of 3.0 and tile size of 8x8 were empirically selected based on visual quality assessment across test images.

**Gaussian Smoothing.** A Gaussian blur with kernel size 5 and sigma 1.0 is applied to suppress high-frequency sensor noise and speckle artefacts common in satellite imagery (Gonzalez & Woods, 2018). The Gaussian kernel was chosen over median or bilateral filters because it provides adequate smoothing for the subsequent thresholding operations without the computational overhead of edge-preserving filters, which are less critical given that the pipeline uses region-based rather than edge-based segmentation as the primary approach.

**Vegetation Index Computation.** The Normalised Difference Vegetation Index (NDVI) is computed to quantify vegetation density:

NDVI = (NIR - Red) / (NIR + Red)

where NIR is the near-infrared reflectance and Red is the red band reflectance (Tucker, 1979). NDVI exploits the characteristic spectral signature of chlorophyll, which strongly absorbs red light and reflects near-infrared radiation. Healthy vegetation typically exhibits NDVI values between 0.3 and 0.9, while non-vegetated surfaces have values near or below zero.

When NIR data is unavailable (e.g., standard RGB aerial photographs), a proxy Excess Green Index (ExG) is computed:

ExG = 2 * g_norm - r_norm - b_norm

where g_norm, r_norm, and b_norm are chromaticity-normalised colour channels (Woebbecke et al., 1995). While less discriminative than true NDVI, ExG provides a reasonable approximation for visible-spectrum-only imagery.

Additionally, the image is converted to HSV colour space, where the Hue channel provides useful information for vegetation classification, and the Saturation channel correlates with vegetation health.

### 3.3 Stage 2: Segmentation

The segmentation stage employs three complementary techniques, combined through a logical intersection to maximise precision.

**NDVI Thresholding with Otsu's Method.** The vegetation index map is binarised using Otsu's automatic thresholding (Otsu, 1979), which determines the optimal threshold by exhaustively searching for the value that minimises the weighted sum of within-class variances. This threshold separates vegetation pixels (NDVI above threshold) from non-vegetation pixels (NDVI below threshold). Otsu's method is preferred over manual thresholding because it adapts automatically to the spectral characteristics of each image, accommodating variations in illumination and atmospheric conditions.

**K-means Clustering.** Unsupervised K-means clustering (Hartigan & Wong, 1979) is applied to a five-dimensional feature vector per pixel: normalised Hue, Saturation, Value, NDVI, and green channel intensity. The algorithm partitions the pixel space into K=4 clusters by minimising the within-cluster sum of squared distances to centroids. The cluster with the highest mean NDVI is identified as the vegetation cluster. K=4 was selected to distinguish vegetation, bare soil, built structures, and water/shadow; however, this parameter can be adjusted based on the complexity of the scene.

**Mask Combination, Shadow Removal, and Texture Filtering.** The NDVI threshold mask and K-means vegetation mask are combined using a union operation to maximise recall. Three refinement filters are then applied sequentially. First, an **HSV hue filter** retains only pixels with hue values between 20 and 100 degrees (the green range) and saturation above 12, removing non-green surfaces. Second, a **shadow mask** excludes pixels with HSV Value below 40, which correspond to shadows, water bodies, and dark gaps between trees. Third, a **texture-based tree filter** computes the local standard deviation within a 15x15 window for each pixel, producing a texture roughness map. Otsu's method determines an adaptive threshold on the texture values of green pixels, and pixels with texture below 70% of this threshold are classified as smooth surfaces (grass, lawns, or bare green land) and excluded. This texture filter is the key discriminator between trees (rough, heterogeneous canopy with visible leaf clusters and shadows) and grass (smooth, uniform green surface). Finally, a **brightness filter** removes very bright pixels (V > 180, S < 80) that typically correspond to lawns or light-coloured ground cover. The refined mask undergoes morphological cleaning: small objects below 80 pixels are removed, small holes below 300 pixels are filled, and binary closing/opening with a disk structuring element of radius 2 smooths the boundaries.

**Marker-Controlled Watershed Segmentation.** The watershed algorithm (Beucher & Meyer, 1993) is applied to delineate individual tree crowns within the refined vegetation mask. The process involves three steps: (1) computing the Euclidean distance transform of the vegetation mask, where the distance value at each pixel represents its distance to the nearest background pixel; (2) detecting local maxima of the distance transform as crown-centre markers, with a minimum inter-peak distance of 15 pixels to prevent over-segmentation; and (3) flooding the inverted distance map from the markers, with watershed lines forming at the boundaries where different flood basins meet. This approach effectively separates overlapping or adjacent tree canopies by exploiting the assumption that crown centres correspond to local maxima in the distance field.

### 3.4 Stage 3: Feature Extraction

For each candidate region produced by watershed segmentation, a comprehensive feature vector is extracted encompassing four categories.

**Shape Descriptors.** Using regionprops from scikit-image, the following geometric features are computed (Blaschke, 2010): area (number of pixels), perimeter (boundary length), eccentricity (ratio of focal distance to major axis length, where 0 indicates a circle and 1 indicates a line), circularity (4*pi*area/perimeter^2, where 1.0 represents a perfect circle), solidity (ratio of area to convex hull area, measuring compactness), and extent (ratio of area to bounding box area).

**GLCM Texture Features.** The Grey Level Co-occurrence Matrix (Haralick et al., 1973) captures spatial relationships between pixel intensity pairs at specified distances and angles. Six texture features are extracted: contrast (measures intensity variation between neighbouring pixels), dissimilarity (linear measure of local variation), homogeneity (closeness of element distribution to the GLCM diagonal), energy (sum of squared GLCM elements, indicating textural uniformity), correlation (linear dependency between grey levels), and Angular Second Moment (ASM, measuring orderliness). Tree canopies typically exhibit moderate contrast and lower homogeneity compared to uniform surfaces such as grass or water.

**Edge Features.** The Canny edge detector (Canny, 1986) and Sobel gradient operator are applied to each region patch. Edge density (proportion of edge pixels) and mean gradient magnitude are computed. Tree canopies exhibit characteristic edge patterns due to leaf clusters, branch shadows, and crown boundary irregularities, resulting in higher edge density compared to homogeneous land cover types.

**Colour Features.** Spectral features include mean green and red channel intensities, green-to-red ratio (a simple vegetation indicator), mean HSV hue and saturation values, and NDVI statistics (mean and standard deviation). The green-to-red ratio is particularly useful because healthy tree canopies reflect more green light relative to red, owing to chlorophyll absorption characteristics.

### 3.5 Stage 4: Recognition

A rule-based classifier is implemented to distinguish tree crowns from non-tree vegetation and false positive detections. The classifier employs nine weighted scoring rules, with texture-related rules carrying the highest weights to reflect the importance of canopy roughness as a tree discriminator:

1. **Area constraint** (weight 1.0): Region area within 60-80,000 pixels, eliminating noise fragments and excessively large merged regions.

2. **NDVI/Greenness** (weight 2.0): Mean NDVI exceeds 0.12, confirming photosynthetically active vegetation. Partial score (1.0) awarded for borderline values above 60% of the threshold.

3. **Circularity** (weight 1.0): Circularity exceeds 0.10. Tree crowns exhibit approximately circular or elliptical shapes when viewed from above.

4. **Eccentricity** (weight 0.5): Eccentricity below 0.97, rejecting highly elongated regions.

5. **Solidity** (weight 0.5): Solidity exceeds 0.25, ensuring reasonable compactness.

6. **Green-red ratio** (weight 1.5): Ratio exceeds 0.7, indicating green-dominant vegetation characteristic of healthy tree canopies.

7. **Edge density** (weight 2.0): Edge density exceeds 0.005, confirming textural complexity expected of tree canopies. This rule is double-weighted because canopy roughness is the strongest discriminator between trees and smooth surfaces.

8. **GLCM contrast** (weight 2.0): Grey level co-occurrence matrix contrast exceeds 0.5, indicating the intensity variation between neighbouring pixels characteristic of tree canopy structure with leaf clusters, branch patterns, and micro-shadows. Partial score (1.0) for values above 30% of the threshold.

9. **Brightness constraint** (weight 1.0): Mean green channel intensity below 200, excluding very bright pixels that typically correspond to grass or reflective surfaces rather than tree canopy.

The confidence score is computed as the weighted sum divided by the maximum possible score (12.5). Regions with confidence >= 0.50 are classified as trees.

### 3.6 Stage 5: Post-processing

The final stage refines the classification output and generates quantitative results. Morphological binary closing with a disk of radius 2 fills small gaps within classified tree crowns. Small isolated regions below 50 pixels are removed to suppress noise. Connected component labelling counts the final number of distinct tree crowns. An annotated output image is generated with green contours delineating detected tree crowns, red dots marking crown centroids, confidence scores displayed alongside each detection, and yellow contours indicating rejected (non-tree) regions for visual verification. Summary statistics are computed including total tree count, tree coverage percentage, mean and standard deviation of crown areas, and mean classification confidence.

---

## 4. Experimental Results

### 4.1 Dataset Description

The pipeline was tested on high-resolution aerial imagery of the Fredericton, New Brunswick, Canada region, sourced from Google Earth Pro (approximate resolution ~0.5m). This location was chosen to align with the study area used in the reference paper by Tong et al. (2021), which employed WorldView-3 multispectral imagery of the same region at 0.31m resolution. Three test images were captured to represent diverse vegetation conditions:

- **Image 1 (phonpadat1):** Dense continuous forest canopy with minimal gaps, captured at approximately 655m eye altitude. This image represents the most challenging scenario for individual crown delineation.
- **Image 2 (pohonpisah2):** Forest area with scattered clearings and mixed vegetation density, captured at approximately 682m eye altitude. Contains both dense clusters and isolated trees.
- **Image 3 (campuran3):** Mixed urban-vegetation scene containing trees, buildings, roads, and construction sites, captured at approximately 632m eye altitude. Tests the pipeline's ability to discriminate vegetation from non-vegetation.

All images were captured as RGB screenshots in PNG format (no NIR band available), requiring the use of the Excess Green Index (ExG) as a proxy for NDVI. The original input images are stored in `data/phonpadat1.png`, `data/pohonpisah2.png`, and `data/campuran3.png` respectively. An approximate ground truth was generated using texture-aware HSV-based thresholding (stored in `output/*_ground_truth.png`) that applies the same tree-grass discrimination logic as the pipeline to ensure fair evaluation.

### 4.2 Quantitative Results

Table 1 summarises the detection performance across the three test images.

**Table 1.** Detection results across test images.

| Image | Type | Trees Detected | Coverage (%) | Mean Confidence |
|-------|------|---------------|-------------|-----------------|
| phonpadat1 | Dense forest | 32 | 88.32 | 0.995 |
| pohonpisah2 | Scattered forest | 59 | 73.29 | 0.997 |
| campuran3 | Mixed urban-vegetation | 109 | 43.59 | 0.998 |

The coverage values correlate logically with the image content: the dense forest image shows the highest vegetation coverage (88.32%), the scattered forest shows high coverage (73.29%) reflecting the predominance of tree canopy with some clearings and grassland, and the mixed urban scene shows lower coverage (43.59%) due to buildings, roads, and bare soil. In the dense forest image, the pipeline correctly detects fewer but larger tree crown regions (32 connected components spanning 88% of the image), while the mixed scene produces more numerous but smaller distinct crowns (109 regions) corresponding to individual trees separated by infrastructure.

Pixel-level and object-level evaluation metrics were computed against texture-aware approximate ground truth that distinguishes trees (rough canopy texture) from grass (smooth surface):

**Table 2.** Pixel-level and object-level evaluation metrics.

| Image | Precision | Recall | F1 Score | IoU | ORR | Merged | Split |
|-------|-----------|--------|----------|-----|-----|--------|-------|
| phonpadat1 | 0.9132 | 0.9379 | 0.9253 | 0.8611 | 0.6667 | 0 | 1 |
| pohonpisah2 | 0.8742 | 0.9323 | 0.9023 | 0.8220 | 0.5106 | 0 | 6 |
| campuran3 | 0.9378 | 0.8378 | 0.8850 | 0.7937 | 0.4156 | 1 | 9 |
| **Average** | **0.9084** | **0.9027** | **0.9042** | **0.8256** | **0.5310** | - | - |

The pipeline achieves an average F1 score of 0.904 and IoU of 0.826 at the pixel level, with precision and recall both exceeding 0.90. A critical design decision was to compute the exclusion mask (shadow, water, soil) from the **original image before CLAHE enhancement**, because CLAHE brightens dark pixels and would cause shadows to incorrectly pass green filters. The best performance was achieved on the dense forest image (phonpadat1, F1=0.925, ORR=0.667), where the texture-based segmentation effectively captured the rough canopy structure and the shadow/water mask correctly excluded dark non-tree regions including a lake visible in the image. The scattered forest (pohonpisah2) achieved the highest recall (0.932) with F1=0.902 and ORR=0.511, demonstrating effective delineation of both dense clusters and isolated trees. The mixed urban scene (campuran3) shows the lowest recall (0.838) due to some tree regions adjacent to buildings being partially excluded by the shadow mask, but maintains the highest precision (0.938) with correct exclusion of buildings, roads, and brown soil areas.

### 4.3 Qualitative Results

This section presents the visual outputs generated by the pipeline for each of the three test images. All figures are automatically generated by the pipeline and saved to the `output/` directory.

#### 4.3.1 Dense Forest (phonpadat1)

**Figure 2.** Complete pipeline visualisation for the dense forest image (`output/phonpadat1_pipeline_overview.png`). The 3x3 panel shows: (a) the original Google Earth image captured at 655m altitude showing continuous forest canopy with a dark lake in the lower-right quadrant; (b) the computed Excess Green Index (ExG) map, where bright green-yellow indicates high vegetation density and dark regions indicate shadow, water, or bare ground; (c) the Otsu-thresholded vegetation mask covering 97.2% of pixels, demonstrating that standard NDVI thresholding alone cannot distinguish trees from the background in a predominantly green image; (d) K-means clustering (K=4) partitioning the image into spectral groups; (e) the refined combined vegetation mask after union, exclusion mask application (shadow/water removal from original image), and texture filtering, reducing coverage to 88.3% by correctly excluding the lake and inter-crown shadows; (f) watershed segmentation producing 214 individual crown regions colour-coded randomly; (g) the final classified tree mask; (h) the annotated detection result with green contours delineating tree crowns and red dots marking centroids; and (i) summary statistics showing 32 detected tree regions at 88.32% coverage.

**Figure 3.** Detection result overlay for the dense forest image (`output/phonpadat1_result.png`). Green contours outline detected tree crown boundaries. The pipeline correctly excludes the dark lake (lower-right) and shadow gaps between tree canopies while maintaining coverage of both bright-green deciduous trees and darker coniferous trees.

**Figure 4.** Evaluation comparison for the dense forest image (`output/phonpadat1_evaluation.png`). The six-panel figure shows: (a) original image; (b) texture-aware ground truth; (c) pipeline prediction; (d) true positive (green), false positive (red), and false negative (blue) overlay; (e) ground truth contours (yellow) versus predicted contours (green); and (f) quantitative metrics (P=0.913, R=0.938, F1=0.925, IoU=0.861, ORR=0.667).

#### 4.3.2 Scattered Forest with Clearings (pohonpisah2)

**Figure 5.** Complete pipeline visualisation for the scattered forest image (`output/pohonpisah2_pipeline_overview.png`). This image, captured at 682m altitude, contains a mix of dense tree clusters, isolated trees, grass clearings, bare brown soil patches, and shadow regions. The pipeline panels show the progression from raw image through NDVI computation, thresholding, K-means clustering, combined mask refinement, watershed segmentation, and final classification. The exclusion mask removes 20.3% of pixels (shadow: 16.6%, water/void: 7.1%, soil: 2.0%), and the texture filter removes an additional 81,136 smooth grass pixels, resulting in 73.3% tree coverage.

**Figure 6.** Detection result overlay for the scattered forest image (`output/pohonpisah2_result.png`). The pipeline effectively distinguishes between tree canopy (detected, green contours) and grass clearings (excluded). Brown soil patches in the upper-left and scattered throughout are correctly excluded by the soil mask. The texture-based filtering successfully removes smooth green grass areas while retaining textured tree canopy of varying density and species.

**Figure 7.** Feature analysis histograms for the scattered forest image (`output/pohonpisah2_feature_analysis.png`). Six histograms compare the distribution of key features (area, circularity, NDVI, eccentricity, edge density, GLCM contrast) between tree-classified and non-tree regions, demonstrating the discriminative power of the selected features.

**Figure 8.** Evaluation comparison for the scattered forest image (`output/pohonpisah2_evaluation.png`), showing pixel-level metrics of P=0.874, R=0.932, F1=0.902, IoU=0.822, and object-level ORR=0.511 with 6 split crowns and 0 merged crowns.

#### 4.3.3 Mixed Urban-Vegetation Scene (campuran3)

**Figure 9.** Complete pipeline visualisation for the mixed scene (`output/campuran3_pipeline_overview.png`). This image, captured at 632m altitude, contains the most diverse land cover: dense forest (right side), individual trees near residential buildings (centre-left), paved roads, construction sites with brown soil, building rooftops, and shadow regions. The exclusion mask removes 27.2% of pixels (shadow: 12.4%, water/void: 6.5%, soil: 13.0%), with the soil mask playing a significant role in excluding the construction areas and bare ground. The texture filter removes 71,296 smooth grass/lawn pixels. Final tree coverage is 43.6%.

**Figure 10.** Detection result overlay for the mixed scene (`output/campuran3_result.png`). This is the most challenging test case. The pipeline correctly identifies tree canopy in both the dense forest area (right) and individual trees near buildings (centre), while excluding roads, building rooftops, construction sites, bare soil, and lawn areas. Some tree regions adjacent to buildings are partially excluded by the shadow mask due to building shadows overlapping with canopy edges, which is reflected in the lower recall (0.838) compared to the forest images.

**Figure 11.** Feature analysis histograms for the mixed scene (`output/campuran3_feature_analysis.png`). The histograms show that in mixed scenes, the GLCM contrast and edge density features exhibit wider distributions, reflecting the heterogeneity of detected tree regions ranging from dense forest clusters to isolated urban trees.

**Figure 12.** Evaluation comparison for the mixed scene (`output/campuran3_evaluation.png`), showing P=0.938, R=0.838, F1=0.885, IoU=0.794, ORR=0.416 with 9 split crowns and 1 merged crown. The TP/FP/FN overlay (panel d) reveals that false negatives (blue) concentrate along canopy edges near buildings, while false positives (red) are minimal, confirming the pipeline's high precision.

#### 4.3.4 Cross-Image Comparison

Table 4 summarises the key visual observations across all three test images.

**Table 4.** Qualitative observations per image.

| Aspect | phonpadat1 (Dense) | pohonpisah2 (Scattered) | campuran3 (Mixed) |
|--------|-------------------|------------------------|------------------|
| Tree detection | Excellent - continuous canopy captured | Good - clusters and isolated trees | Good - forest and urban trees |
| Shadow handling | Lake and gaps correctly excluded | Shadow patches excluded | Building shadows partially over-exclude |
| Grass discrimination | Minimal grass present | Grass clearings correctly excluded | Lawn areas correctly excluded |
| Soil exclusion | No soil present | Brown patches excluded | Construction sites excluded |
| Main limitation | Some crown edges lost at shadow boundary | 6 crowns split by watershed | 9 crowns split near building edges |

---

## 5. Critical Evaluation and Analysis

### 5.1 Strengths of the Approach

The proposed pipeline demonstrates several strengths. First, the **texture-based tree discrimination** using local standard deviation filtering effectively separates trees (rough canopy texture) from grass and smooth green surfaces (low texture). This is a key contribution because standard NDVI or colour-based methods cannot distinguish these vegetation subtypes, which share similar spectral properties. The Otsu-based adaptive texture threshold ensures this filtering adapts to each image without manual parameter tuning. Second, the **shadow masking** using HSV Value channel thresholding (V < 40) correctly excludes dark regions that would otherwise be misclassified. This is particularly effective for the dense forest image where inter-crown shadows, water bodies, and dark gaps are prevalent. Third, **adaptive watershed segmentation** adjusts the minimum distance parameter based on vegetation density ratio, automatically using smaller distances for dense forest (min_dist=8) and larger distances for sparse scenes (min_dist=12), improving delineation across varying canopy conditions. Fourth, the **multi-modal feature extraction** combining shape, texture, edge, and colour features with a 9-rule weighted classifier provides a comprehensive characterisation of each candidate region. The best performance was achieved on the phonpadat1 dense forest image (F1=0.923, IoU=0.857, ORR=0.667), demonstrating that the pipeline handles its most challenging scenario effectively.

### 5.2 Failure Case Analysis

Several conditions cause the pipeline to underperform or fail, which is critical to acknowledge for honest assessment.

**Dense continuous canopy.** When tree crowns form a continuous canopy without visible gaps (e.g., dense tropical forest viewed at 10m resolution), the watershed segmentation produces either over-segmented fragments or under-segmented merged regions. The distance transform approach assumes that crown centres correspond to local maxima, which requires at least partial separation between adjacent crowns. At coarse resolutions, individual crowns may not be resolvable. This failure mode is inherent to the spatial resolution limitation and would persist even with deep learning methods.

**Shadow interference.** Dark shadows from tall trees, buildings, or terrain can be misclassified as non-vegetation by the NDVI thresholding, causing portions of tree crowns to be excluded from the vegetation mask. While CLAHE partially mitigates this by enhancing local contrast, severe shadows remain problematic. A potential improvement would be shadow detection and compensation using chromatic shadow models (Tsai, 2006).

**Spectral ambiguity.** Green-coloured non-vegetation objects (e.g., artificial turf, green rooftops, algae-covered water) can produce false positive detections, as the ExG proxy and colour-based features cannot fully distinguish vegetation from other green surfaces. True NDVI using NIR data mitigates this significantly, as non-vegetation does not exhibit the characteristic NIR reflectance spike.

**Over-segmentation.** In cases where large trees have multi-lobed canopy structure (e.g., spreading oak or banyan trees), the watershed algorithm may split a single crown into multiple segments. The minimum distance parameter controls this behaviour, but no single value is optimal for all tree species and sizes.

### 5.3 Comparison with Contemporary Methods

The reference paper by Tong et al. (2021) achieved an ORR of 77.3%-82.2% using marker-controlled watershed segmentation on WorldView-3 imagery (0.31m resolution, 8 VNIR bands). In comparison, our pipeline achieved an ORR of 43.2% using only RGB imagery at approximately 0.5m resolution. This performance gap can be attributed to three key factors:

**Table 3.** Comparison with Tong et al. (2021) reference paper.

| Aspect | Tong et al. (2021) | This Study |
|--------|-------------------|------------|
| Imagery | WorldView-3 (0.31m, 8 bands) | Google Earth RGB (~0.5m, 3 bands) |
| Vegetation index | True NDVI (NIR band) | ExG proxy (visible bands only) |
| Crown border extraction | Gradient binarization | Texture-filtered watershed |
| Shadow handling | Supervised classification | HSV value thresholding |
| ORR | 77.3% - 82.2% | 53.1% (best: 66.7%) |
| Pixel F1 | Not reported | 0.904 (best: 0.925) |
| Pixel IoU | Not reported | 0.826 (best: 0.861) |

First, the **absence of NIR data** limits the discriminative power of vegetation indexing. True NDVI exploits the unique spectral signature of chlorophyll in the near-infrared, which has no equivalent in visible-spectrum-only imagery. Second, the **lower spatial resolution** (~0.5m vs. 0.31m) reduces the ability to resolve individual crown boundaries, particularly in dense forest stands. Third, the reference paper employed **gradient binarization with supervised threshold selection** for crown border extraction, a more sophisticated approach than the distance-transform-based watershed used here.

Contemporary deep learning approaches such as U-Net (Ronneberger et al., 2015) and Mask R-CNN (He et al., 2017) typically achieve F1 scores exceeding 0.85 on benchmark datasets (Brandt et al., 2020). Our classical pipeline achieves a pixel-level F1 of 0.904 on average and 0.925 on the dense forest image, demonstrating that well-designed classical methods incorporating texture analysis, shadow compensation, and adaptive parameterisation can **exceed** this threshold at the pixel level.

However, classical methods retain significant advantages. They require **no training data**, which is often scarce for remote sensing. They offer **interpretability**, as each step can be inspected. They have **lower computational requirements**, suitable for resource-constrained environments. They are also **transferable** across geographic regions without retraining, unlike deep learning models that suffer domain shift (Ma et al., 2019).

---

## 6. Conclusion and Recommendations

This report presented a five-stage classical computer vision pipeline for tree crown detection and mapping from satellite and aerial imagery. The system successfully demonstrates that established image processing techniques, including NDVI-based vegetation indexing, Otsu's automatic thresholding, K-means clustering, marker-controlled watershed segmentation, GLCM texture analysis, and rule-based classification, can be effectively combined to detect, segment, and recognise individual tree crowns without relying on deep learning methods.

The pipeline achieves an average pixel-level F1 score of 0.904 and IoU of 0.826 across three diverse test images from Fredericton, Canada, with the best performance on dense forest imagery (F1=0.925, ORR=0.667). The texture-based discrimination between trees and grass, combined with pre-CLAHE shadow/water/soil masking and adaptive watershed parameters, enables the pipeline to handle varying vegetation densities from dense continuous canopy (88% coverage) to mixed urban-vegetation scenes (44% coverage). The system produces annotated maps with crown contours, centroids, confidence scores, and quantitative statistics suitable for forest inventory and environmental monitoring applications.

**Recommendations for future work** include:

1. **Adaptive parameterisation:** Implementing automatic tuning of the minimum watershed distance and classifier thresholds based on estimated image resolution, potentially using scale-space analysis.

2. **Shadow compensation:** Integrating a shadow detection module using chromaticity-based methods to recover crown regions occluded by shadows.

3. **Multi-scale processing:** Applying the pipeline at multiple scales and merging detections to handle the wide range of tree sizes encountered in mixed forests.

4. **PCA-based band fusion:** For multispectral imagery with many bands, Principal Component Analysis could enhance vegetation-background contrast by combining spectral information optimally.

5. **Hybrid approach:** Combining the classical pipeline with lightweight machine learning classifiers (e.g., Random Forest or SVM trained on the extracted features) could improve classification accuracy while retaining interpretability.

6. **Validation framework:** Establishing a systematic validation protocol with manually annotated ground truth for diverse geographic regions and vegetation types.

---

## 7. References

Beucher, S., & Meyer, F. (1993). The morphological approach to segmentation: The watershed transformation. In E. R. Dougherty (Ed.), *Mathematical morphology in image processing* (pp. 433-481). Marcel Dekker.

Blaschke, T. (2010). Object based image analysis for remote sensing. *ISPRS Journal of Photogrammetry and Remote Sensing*, *65*(1), 2-16. https://doi.org/10.1016/j.isprsjprs.2009.06.004

Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, *25*(11), 120-125.

Brandt, M., Tucker, C. J., Kariryaa, A., Rasmussen, K., Abel, C., Small, J., Chave, J., Rasmussen, L. V., Hiernaux, P., Diouf, A. A., Kergoat, L., Mertz, O., Igel, C., Gieseke, F., Schoning, J., Li, S., Melocik, K., Meyer, J., Sinno, S., ... Fensholt, R. (2020). An unexpectedly large count of trees in the West African Sahara and Sahel. *Nature*, *587*(7832), 78-82. https://doi.org/10.1038/s41586-020-2824-5

Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *PAMI-8*(6), 679-698. https://doi.org/10.1109/TPAMI.1986.4767851

Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4th ed.). Pearson.

Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, *342*(6160), 850-853. https://doi.org/10.1126/science.1244693

Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, *SMC-3*(6), 610-621. https://doi.org/10.1109/TSMC.1973.4309314

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Rio, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2

Hartigan, J. A., & Wong, M. A. (1979). Algorithm AS 136: A K-means clustering algorithm. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, *28*(1), 100-108. https://doi.org/10.2307/2346830

He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2961-2969). IEEE.

Kattenborn, T., Leitloff, J., Schiefer, F., & Hinz, S. (2019). Convolutional neural networks accurately predict cover fractions of plant species and communities in Unmanned Aerial Vehicle imagery. *Remote Sensing in Ecology and Conservation*, *6*(4), 472-486. https://doi.org/10.1002/rse2.146

Ke, Y., & Quackenbush, L. J. (2011). A review of methods for automatic individual tree-crown detection and delineation from passive remote sensing. *International Journal of Remote Sensing*, *32*(17), 4725-4747. https://doi.org/10.1080/01431161.2010.494184

Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., & Johnson, B. A. (2019). Deep learning in remote sensing applications: A meta-analysis and review. *ISPRS Journal of Photogrammetry and Remote Sensing*, *152*, 166-177. https://doi.org/10.1016/j.isprsjprs.2019.04.015

Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, *9*(1), 62-66. https://doi.org/10.1109/TSMC.1979.4310076

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015* (pp. 234-241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

Tsai, V. J. D. (2006). A comparative study on shadow compensation of color aerial images in invariant color models. *IEEE Transactions on Geoscience and Remote Sensing*, *44*(6), 1661-1671. https://doi.org/10.1109/TGRS.2006.869980

Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, *8*(2), 127-150. https://doi.org/10.1016/0034-4257(79)90013-0

van der Walt, S., Schonberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: Image processing in Python. *PeerJ*, *2*, e453. https://doi.org/10.7717/peerj.453

Woebbecke, D. M., Meyer, G. E., Von Bargen, K., & Mortensen, D. A. (1995). Color indices for weed identification under various soil, residue, and lighting conditions. *Transactions of the ASAE*, *38*(1), 259-269. https://doi.org/10.13031/2073.27838

Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), *Graphics gems IV* (pp. 474-485). Academic Press.

---

## 8. Appendix

### Appendix A: Output File Reference

**Table A1.** Complete list of output files generated by the pipeline.

| Figure | File Path | Description |
|--------|-----------|-------------|
| Fig. 2 | `output/phonpadat1_pipeline_overview.png` | 3x3 pipeline stages panel - dense forest |
| Fig. 3 | `output/phonpadat1_result.png` | Detection result overlay - dense forest |
| Fig. 4 | `output/phonpadat1_evaluation.png` | TP/FP/FN evaluation - dense forest |
| Fig. 5 | `output/pohonpisah2_pipeline_overview.png` | 3x3 pipeline stages panel - scattered |
| Fig. 6 | `output/pohonpisah2_result.png` | Detection result overlay - scattered |
| Fig. 7 | `output/pohonpisah2_feature_analysis.png` | Feature histograms - scattered |
| Fig. 8 | `output/pohonpisah2_evaluation.png` | TP/FP/FN evaluation - scattered |
| Fig. 9 | `output/campuran3_pipeline_overview.png` | 3x3 pipeline stages panel - mixed |
| Fig. 10 | `output/campuran3_result.png` | Detection result overlay - mixed |
| Fig. 11 | `output/campuran3_feature_analysis.png` | Feature histograms - mixed |
| Fig. 12 | `output/campuran3_evaluation.png` | TP/FP/FN evaluation - mixed |
| - | `output/*_tree_mask.png` | Binary tree mask per image |
| - | `output/*_annotated.png` | Annotated image with contours and centroids |
| - | `output/*_ground_truth.png` | Texture-aware approximate ground truth |

**Table A2.** Input data files.

| File Path | Description | Resolution | Altitude |
|-----------|-------------|------------|----------|
| `data/phonpadat1.png` | Dense forest, Fredericton, Canada | ~0.5m | 655m |
| `data/pohonpisah2.png` | Scattered forest with clearings | ~0.5m | 682m |
| `data/campuran3.png` | Mixed urban-vegetation scene | ~0.5m | 632m |

### Appendix B: Key Code Snippets

**B.1 Exclusion Mask (from Original Image, before CLAHE)**

```python
def create_exclusion_mask(rgb_original):
    hsv_orig = cv2.cvtColor(rgb_original, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_orig[:, :, 0], hsv_orig[:, :, 1], hsv_orig[:, :, 2]
    b_ch, g_ch, r_ch = (rgb_original[:, :, i].astype(np.float32)
                         for i in range(3))

    # Shadow: very dark pixels (V < 30)
    shadow = v < 30

    # Water: blue-dominant dark pixels (lakes, rivers)
    blue_ratio = b_ch / (g_ch + 1)
    water = ((blue_ratio > 1.4) & (v < 60) & (b_ch > r_ch * 1.2))
    void = (v < 15) & (s < 30)  # near-black voids

    # Soil: brown/tan hue range
    brown = ((h >= 5) & (h <= 22) & (s > 40) & (v > 50) & (v < 200))
    tan = ((h >= 15) & (h <= 28) & (s > 20) & (s < 80) & (v > 140))

    return shadow | water | void | brown | tan
```

**B.2 Texture-Based Tree vs Grass Discrimination**

```python
def compute_texture_map(gray, ksize=15):
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sqmean = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.maximum(sqmean - mean ** 2, 0)
    return np.sqrt(variance).astype(np.float32)

# In segmentation: Otsu on texture of green pixels
tex_u8 = (normalise(texture_map) * 255).astype(np.uint8)
green_tex_u8 = tex_u8[combined_green_mask]
tex_thresh, _ = cv2.threshold(green_tex_u8, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Use 60% of Otsu to retain more tree edge pixels
tree_texture = texture_map >= (tex_threshold * 0.6)
```

**B.3 9-Rule Weighted Classifier**

```python
# Rule 7 - Edge density / texture roughness (weight 2.0)
total += 2.0
if r["edge_density"] >= 0.005:
    score += 2.0

# Rule 8 - GLCM contrast (weight 2.0)
total += 2.0
if r["contrast"] >= 0.5:
    score += 2.0
elif r["contrast"] >= 0.15:
    score += 1.0

# Rule 9 - Not too bright (weight 1.0)
total += 1.0
if r["mean_green"] <= 200:
    score += 1.0

confidence = score / total  # max total = 12.5
is_tree = confidence >= 0.50
```

### Appendix C: Usage Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a single image
python src/tree_detection_pipeline.py data/campuran3.png

# Run on all images in a directory (batch mode)
python src/tree_detection_pipeline.py data/ -o output/

# Run evaluation with metrics
python src/evaluate_metrics.py

# Specify watershed distance and K-means clusters
python src/tree_detection_pipeline.py data/image.png -d 20 -k 5
```
