# Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images
Python implementation of shadow detection and correction algorithm proposed in academia

This repository contains functions to detect shadow - covered areas in aerial/satellite imagery and to correct for the brightness of 
the image in the shadow - covered areas as proposed in the paper 'Near Real - Time Shadow Detection and Removal in Aerial Motion 
Imagery Application' by Silva G.F., Carneiro G.B., Doth R., Amaral L.A., de Azevedo D.F.G. (2017).

In addition, the multilevel Otsu thresholding method is not used in view of the computational complexity and time cost incurred. 
As such, it is replaced by the K - Means clustering algorithm, which is another method to determine the multiple thresholds in a global context, which would give a close approximation to that of the multilevel Otsu thresholding method.

Requirements:
- gc
- numpy
- pandas
- rasterio
- scikit - image
- scikit - learn
