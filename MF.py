import cv2
from skimage.filters import median
from skimage.morphology import disk

src = cv2.imread("images/a.png")
img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
skimedianimg = median(img,disk(3))


cv2.imshow("blurred",skimedianimg)
cv2.imshow("original",src)


cv2.waitKey(0)
cv2.destroyAllWindows()