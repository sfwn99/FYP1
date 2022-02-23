
import cv2
from skimage.filters import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from scipy import ndimage
from PIL import Image
import os



inPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\images\\DTOP"
outPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\segmented"

imagePath = 'D-13T.jpg'
inputPath = os.path.join(inPath, imagePath)

#Read image, cvt from colored to gray, and perform blur
src = cv2.imread(inputPath)
img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img_blur = median(img,disk(2))

#Perform Canny Edge detection
t = 9
detected_edges = cv2.Canny(img_blur,t,t*3,3)


#Dilate the edges
kernel1 = np.ones((4,4),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
erosion = cv2.erode(detected_edges,kernel2,iterations = 1)
closing = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel2)
mask_dilate = cv2.dilate(closing,kernel1,iterations=1)



#Create Mask
mask = mask_dilate != 0
dst = src * (mask[:,:,None].astype(src.dtype))
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
result = np.zeros_like(img)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

fullOutPath = os.path.join(outPath, 'segmented_' + imagePath)

cv2.imwrite(fullOutPath, cv2.bitwise_and(result, img))



cv2.imshow("Source",img)
cv2.imshow("Filled",result)
cv2.imshow("Detected",detected_edges)
cv2.imshow("Eroded",erosion)
cv2.imshow("Dilate",mask_dilate)
cv2.imshow("Closing",closing)
"""

cv2.imshow("Image",src)
cv2.imshow("Blur",img_blur)
cv2.imshow("Edge Detection",dst)
cv2.imshow("Eroded",mask_dilate)
"""
#Applied Masking
cv2.imshow("Segmented Mango",cv2.bitwise_and(result,img))
cv2.waitKey(0)
cv2.destroyAllWindows()
