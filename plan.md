Pipeline Mapping Pohon (Classical CV)
1. Preprocessing
Band selection: Ambil kanal hijau atau buat indeks vegetasi (NDVI).

Enhancement: Gunakan histogram equalization untuk kontras.

Noise removal: Gaussian blur ringan untuk mengurangi speckle.

2. Segmentation
Thresholding NDVI → pisahkan vegetasi vs non-vegetasi.

Clustering (K-means/ISODATA) → klasifikasi piksel vegetasi.

Watershed segmentation → memisahkan kanopi pohon yang berdekatan.

3. Feature Extraction
Shape descriptors: gunakan regionprops (MATLAB) untuk area, eccentricity, circularity.

Texture analysis: GLCM atau Gabor filter untuk membedakan pohon dari semak/rumput.

Edge detection: Canny/Sobel untuk batas kanopi.

4. Recognition
Rule-based classifier: vegetasi + bentuk bulat + tekstur kasar = pohon.

Template matching: cocokkan pola kanopi bulat.

Morphological filtering: opening/closing untuk memperjelas blob pohon.

5. Post-processing
Connected component labeling → hitung jumlah pohon.

Skeletonization → distribusi spasial pohon.

GIS overlay → integrasi hasil ke peta.




Ide Eksperimen untuk Assigment:

Thresholding NDVI untuk memisahkan vegetasi.

K-means clustering pada kanal hijau untuk klasifikasi pohon vs non-pohon.

Watershed segmentation untuk memisahkan kanopi pohon yang berdekatan.

Morphological opening/closing untuk memperjelas bentuk pohon.

GLCM texture analysis untuk membedakan pohon dari semak/rumput.

Hough Transform untuk mendeteksi bentuk bulat kanopi.

Region growing dari seed vegetasi untuk delineasi pohon.

Template matching dengan kanopi pohon bulat.

Edge detection + contour analysis untuk ekstraksi batas pohon.

PCA-based band fusion untuk meningkatkan kontras vegetasi.

Histogram thresholding untuk water vs vegetation separation (baseline).

Connected component labeling untuk menghitung jumlah pohon.

Skeletonization untuk memetakan distribusi pohon.

Rule-based classifier (warna + tekstur + bentuk).

Gabungan clustering + morphological filtering untuk hasil lebih robust.

