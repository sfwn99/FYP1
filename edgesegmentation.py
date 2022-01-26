import cv2
from skimage.filters import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from scipy import ndimage
from PIL import Image
import os


def main():
    inPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\editedimages\\DTOP"
    outPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\segmented\\DTOP"

    for imagePath in os.listdir(inPath):

        inputPath = os.path.join(inPath, imagePath)

        #Read image, cvt from colored to gray, and perform blur
        src = cv2.imread(inputPath)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img_blur = median(img,disk(1))

        #Perform Canny Edge detection
        t = 12
        detected_edges = cv2.Canny(img_blur,t,t*3,3)

        #Dilate the edges
        kernel = np.ones((2,2),np.uint8)
        mask_dilate = cv2.dilate(detected_edges,kernel,iterations=1)

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

if __name__ == '__main__':
    main()

"""
cv2.imshow("Filled",result)
cv2.imshow("Detected",detected_edges)
cv2.imshow("Source",img)
cv2.imshow("Image",src)
cv2.imshow("Blur",img_blur)
cv2.imshow("Edge Detection",dst)
cv2.imshow("Eroded",mask_dilate)
#Applied Masking
cv2.imshow("Segmented Mango",cv2.bitwise_and(result,img))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
medianimg = median (img,disk(1),mode='constant',cval=0.0)

#Canny Edge Detection

sigma = 1.0
median = np.median(img)
lower = int(max(0,(1.0-sigma)*median))
upper = int(min(255,(1.0+sigma)*median))
img_canny = cv2.Canny(img,lower,upper)

#Fill the detected edge




#cv2.imshow("Medianed",medianimg)
cv2.imshow("AutoCanny",img_canny)
cv2.imshow("Original",img)

#cv2.imwrite("2.png",medianimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""