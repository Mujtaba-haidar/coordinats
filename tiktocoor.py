import cv2
import numpy as np 

def click_event(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X:",x , "Y:",y)
        font = cv2.FONT_HERSHEY_COMPLEX
        s = str(x) + " , " + str(y)

        cv2.putText(img , "*" , (x,y) , font , 0.5, (55,144,144), 1)
        cv2.imshow('image',img)


img = np.zeros((400,720,3),np.uint8)
img = cv2.imread("picturs/window.jpg")
img = cv2.resize(img,(600,600))
cv2.imshow("image",img)

cv2.setMouseCallback('image',click_event )

cv2.waitKey(0)
cv2.destroyAllWindows()