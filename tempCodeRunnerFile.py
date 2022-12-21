import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
img = cv.imread('picturs/window.jpg')
img = cv.resize(img,(600,600))

while True:
    ret,frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    low_red = np.array([0,0,255])
    upp_red = np.array([255,255,255])
    mask = cv.inRange(hsv,low_red,upp_red)

    (minVal , maxVal , minLoc , maxLoc) = cv.minMaxLoc(mask)
    if maxLoc[0] != 0 and maxLoc[1] != 0:
        font= cv.FONT_HERSHEY_COMPLEX
        mask=cv.putText(mask, str(maxLoc[0]) + "," + str(maxLoc[1]) ,maxLoc,font,1,(255,0,0),2,cv.LINE_AA)
        print(maxLoc)
        cv.circle(mask,maxLoc,20,(255,255,255),2,cv.LINE_AA)

        font= cv.FONT_HERSHEY_COMPLEX
        img=cv.putText(img,"*" ,maxLoc,font, 0.5, (0,255,255), 1)
        print(maxLoc)
        #cv.circle(img,maxLoc,20,(255,255,255),2,cv.LINE_AA)
        




    cv.imshow('frame',mask)
    cv.imshow('image',img)

    k=cv.waitKey(1)
    if k == ord('q'):
     break

cap.release()
cv.destroyAllWindows()