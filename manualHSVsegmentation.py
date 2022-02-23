import cv2
import numpy as np
import imutils


src = cv2.imread(r'C:\Users\sfwn9\PycharmProjects\FYP1\images\CTOP\C-1T.jpg')

image = src

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_blur = cv2.GaussianBlur(image, (5, 5), 0)

image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)



minval = np.array([0,0,0])
maxval = np.array([70,255,255])

mask = cv2.inRange(image_blur_hsv, minval, maxval)

dst = src * (mask[:,:,None].astype(src.dtype))
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
result = np.zeros_like(src)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)


cnts = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
contour_sizes = [(cv2.contourArea(contour), contour) for contour in cnts]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
cv2.drawContours(src,[biggest_contour],-1,(0,255,0),1)

area = cv2.contourArea(biggest_contour)
perimeter = cv2.arcLength(biggest_contour,True)

print(area)
print(perimeter)

cv2.imshow("Contour",src)

cv2.waitKey(0)

cv2.destroyAllWindows()