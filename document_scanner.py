import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

image=cv2.imread("/home/ubuntu/pan/05cPCFWrbES3.jpg")   
#resizing the input image because opencv does not work efficiently with large image size
image=cv2.resize(image,(1300,800)) 
orig=image.copy()

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  

th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 1)
# (5,5), 0 is the kernel size and sigma value that determines the amount of blurness in image
blurred=cv2.GaussianBlur(gray,(5,5),0)  
# 30 is the MinThreshold and 50 is the MaxThreshold value
edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold

contours, hierarchy=cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse=True)

# find the boundary contours of object in image
for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapp(target) 

pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  

op=cv2.getPerspectiveTransform(approx,pts) 
dst=cv2.warpPerspective(orig,op,(800,800))

plt.imshow(dst)
plt.show()
plt.imshow(image)
plt.show()
