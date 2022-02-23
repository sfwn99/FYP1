import cv2
import numpy as np
import os

inPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\images\\CTOP"
outPath = "C:\\Users\\sfwn9\\PycharmProjects\\FYP1\\segmented\\CTOP"

for imagePath in os.listdir(inPath):

    inputPath = os.path.join(inPath, imagePath)
    src = cv2.imread(inputPath)
    image = src
    height, width, channels = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    minval = np.array([0,0,0])
    maxval = np.array([60,255,255])

    mask = cv2.inRange(image_blur_hsv, minval, maxval)

    dst = src * (mask[:,:,None].astype(src.dtype))
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    result = np.zeros_like(src)
    cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

    fullOutPath = os.path.join(outPath, 'segmented_' + imagePath)
    cv2.imwrite(fullOutPath, cv2.bitwise_and(result, src))


"""
cv2.imshow("Mask",mask)
cv2.imshow("Segmented Mango",cv2.bitwise_and(result,src))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""