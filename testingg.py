import cv2

src = cv2.imread('segmented/ATOP/segmented_invert_A-1T.jpg')
img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("Picture",img)
cv2.waitKey(0)
cv2.destroyAllWindows()