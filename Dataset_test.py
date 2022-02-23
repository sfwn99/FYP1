# In[1]:

import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np

from imutils import contours
import imutils


# In[2]:


def show(image):
# Figure size in inches
    plt.figure(figsize=(20, 20))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='none')
    
def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)
    
def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
     
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)


# In[20]:


image = cv2.imread(r'C:\Users\sfwn9\PycharmProjects\FYP1\images\DTOP\D-1T.jpg')
assert not isinstance(image,type(None)), 'image not found'

height, width, channels=image.shape

# In[22]:


# Convert from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to a third of the size
image = cv2.resize(image, None, fx=1*3, fy=1*3)
show(image)



# In[24]:

def show_rgb_hist(image):
     colours = ('r','g','b')
     for i, c in enumerate(colours):
         plt.figure(figsize=(20, 4))
         histr = cv2.calcHist([image], [i], None, [256], [0, 256])
         plt.plot(histr, color=c)
        
         if c == 'r': colours = [((i/255, 0, 0)) for i in range(0, 256)]
         if c == 'g': colours = [((0, i/255, 0)) for i in range(0, 256)]
         if c == 'b': colours = [((0, 0, i/255)) for i in range(0, 256)]
        
         plt.xlim([0, 256])

     plt.show()
    
show_rgb_hist(image)

# # In[25]:

def show_hsv_hist(image):

     image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

     colours = ('h','s','v')
    
     plt.figure(figsize=(20, 4))
     histr = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
     plt.plot(histr)
        
     plt.xlim([0, 180])

     plt.show()
    
show_hsv_hist(image)


# In[27]:

# Blur image slightly
image_blur = cv2.GaussianBlur(image, (5, 5), 0)
show(image_blur)

# In[55]:

image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

# 0-10 hue
min_red = np.array([20, 60, 40])
max_red = np.array([80, 255,255])

#min_red = np.array([0, 80, 0])
#max_red = np.array([7, 255, 255])
image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)

show_mask(image_red1)

image_red = image_red1
show_mask(image_red)


# In[56]:

# Clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))

# Fill small gaps
image_red_closed = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel)
show_mask(image_red_closed)

# Remove specks
image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)


#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=10)
#show_mask(opening)

show_mask(image_red_closed_then_opened)

# In[53]:

def find_biggest_contour(image):
    

    # check OpenCV version
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
 
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

big_contour, red_mask = find_biggest_contour(image_red_closed_then_opened)
show_mask(red_mask)

# In[57]:

overlay_mask(red_mask, image) 

# In[58]:

# Bounding ellipse
image_with_ellipse = image.copy()
ellipse = cv2.fitEllipse(big_contour)
cv2.ellipse(image_with_ellipse, ellipse, (0,255,0), 2)
show(image_with_ellipse)

# In[59]:

#  Bounding Box
ret,thresh = cv2.threshold(red_mask,-1,255,-1)

# check OpenCV version
major = cv2.__version__.split('.')[0]
if major == '3':
    ret, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_boundbox=image.copy()

for item in range(len(contours)):
    cnt = contours[item]
    if len(cnt)>20:
        print(len(cnt))
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image_boundbox,(x,y),(x+w,y+h),(0,255,0),2)
        show(image_boundbox)
print (y)
# In[60]:

# Centre of mass

mom = cv2.moments(red_mask)

# Calculating x,y coordinate of center
if mom['m00'] != 0:
    cX = int(mom['m10']/mom['m00'])
    cY = int(mom['m01']/mom['m00'])
else:
    cX,cY = 0, 0

image_with_com = image.copy()
cv2.circle(image_with_com, (cX , cY), 10, (0, 255, 0), -1)
show(image_with_com)

# In[61]:

# extracting perimeter size and area

mom = cv2.moments(red_mask)

perimeter = cv2.arcLength(cnt,True)
area = cv2.contourArea(cnt)

print('The perimeter of the mango is',perimeter)
print('The area of the mango is',area)

# In[63]:

canny = red_mask.copy()
edged = cv2.Canny(canny, 30, 200)

 # check OpenCV version
major = cv2.__version__.split('.')[0]
if major == '3':
    ret, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Canny edges after contour
image_multiple = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

show(image_multiple)
print("Number of mangoes = " + str(len(contours)))


def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i+=1
            
    return largest_area, largest_contour_index

cnt = contours[13]
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
largest_area, largest_contour_index = findGreatesContour(contours)
print(largest_area)
print(largest_contour_index)
print(len(contours))
print(cX)
print(cY)
