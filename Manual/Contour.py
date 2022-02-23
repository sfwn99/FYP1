import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import median

def callback(x):
    print(x)

img = cv2.imread('../images/DTOP/D-13T - Copy.jpg', 0) #read image as grayscale
img_blur = median(img,disk(1))


kernel2 = np.ones((1, 1), np.uint8)
canny = cv2.Canny(img_blur, 85, 255)
closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel2)
cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

while(1):
    numpy_horizontal_concat = np.concatenate((closing, canny), axis=1) # to display image side by side
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(img, l, u)

    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel2)


cv2.destroyAllWindows()