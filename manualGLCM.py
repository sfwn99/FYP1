import matplotlib.pyplot as plt
from skimage.feature import greycomatrix,greycoprops
from skimage import io
import cv2
import numpy as np
from scipy import stats
import matplotlib.patches as patches
def Average(lst):
    return sum(lst)/len(lst)

image = io.imread('editedimages\DTOP\invert_segmented_D-1T.jpg')

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

nsize = (55,100)
#(y,x)
y = 41
x = 75
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

glcm = greycomatrix(region, distances=[1,3],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
corr.append(Average(Average(greycoprops(glcm,'correlation')[:,:])))
energy.append(Average(Average(greycoprops(glcm,'energy')[:,:])))
contrast.append(Average(Average(greycoprops(glcm,'contrast')[:,:])))
ASM.append(Average(Average(greycoprops(glcm,'ASM')[:,:])))
diss.append(Average(Average(greycoprops(glcm,'dissimilarity')[:,:])))
homogene.append(Average(Average(greycoprops(glcm,'homogeneity')[:,:])))




print('Correlation:',corr)
print('Energy:',energy)
print('Contrast:',contrast)
print('ASM:',ASM)
print('Dissimilarity:',diss)
print('Homogeneity:',homogene)