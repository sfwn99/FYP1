import matplotlib.pyplot as plt
from skimage.feature import greycomatrix,greycoprops
from skimage import io
import cv2
import numpy as np

PATCH_SIZE = 35

image = io.imread('4pro.png')

nsize = (220,145)
#(y,x)
region = image[215:215+nsize[0],119:119+nsize[1]]

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(region,cmap='gray')
plt.show()


corr = []
energy = []
contrast = []
ASM = []
glcm = greycomatrix(region, distances=[5],angles=[0], levels = 256, symmetric=True, normed=True)
corr.append(greycoprops(glcm,'correlation')[0,0])
energy.append(greycoprops(glcm,'energy')[0,0])
contrast.append(greycoprops(glcm,'contrast')[0,0])
ASM.append(greycoprops(glcm,'ASM')[0,0])

print('Correlation:',corr)
print('Energy:',energy)
print('Contrast:',contrast)
print('ASM:',ASM)

"""""

man_locations = [(132,231),(200,200)]
scratch_patches = []

for loc in man_locations :
    scratch_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,loc[1]:loc[1] + PATCH_SIZE])


# Compute some GLCM properties each patch
diss_sim = []
corr = []
homogen = []
energy = []
contrast = []

for patch in (cell_patches + scratch_patches):
#Full image
GLCM = greycomatrix(image,[1],[0,np.pi/4,np.pi/2,3*np.pi/4])
a = greycoprops(GLCM,'energy')[0,0]

"""""
