import matplotlib.pyplot as plt
from skimage.feature import greycomatrix,greycoprops
from skimage import io
import cv2
import numpy as np
from scipy import stats
import matplotlib.patches as patches
import timeit

def Average(lst):
    return sum(lst)/len(lst)

image = io.imread('segmented/segmented_invert_1.png')

nsize = (29,29)
#(y,x)
y = 200
x = 100
region = image[y+2:y+nsize[0],x+1:x+nsize[1]]

plt.figure()
ax = plt.subplot(1,2,1)
ax.imshow(image,cmap='gray')
rect = patches.Rectangle((x,y),nsize[1],nsize[0],linewidth=0.5,edgecolor='r',facecolor="none")
ax.add_patch(rect)
ax= plt.subplot(1,2,2)
ax.imshow(region,cmap='gray')
plt.show()

corr = []
energy = []
contrast = []
ASM = []
diss = []
homogene=[]

glcm = greycomatrix(region, distances=[10],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
corr.append(Average(greycoprops(glcm,'correlation')[0,:]))
energy.append(Average(greycoprops(glcm,'energy')[0,:]))
contrast.append(Average(greycoprops(glcm,'contrast')[0,:]))
ASM.append(Average(greycoprops(glcm,'ASM')[0,:]))
diss.append(Average(greycoprops(glcm,'dissimilarity')[0,:]))
homogene.append(Average(greycoprops(glcm,'homogeneity')[0,:]))

print('Correlation:',corr)
print('Energy:',energy)
print('Contrast:',contrast)
print('ASM:',ASM)
print('Dissimilarity:',diss)
print('Homogeneity:',homogene)
