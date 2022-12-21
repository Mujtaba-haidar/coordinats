import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)
img = cv2.imread("picturs/window.jpg")
img = cv2.resize(img,(600,600))
pts = []

while True :
    ret , frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_red = np.array([0,0,255])
    upp_red = np.array([255,255,255])
    mask = cv2.inRange(hsv , low_red , upp_red)

    (minVal , maxVal , minLoc , maxLoc) = cv2.minMaxLoc(mask)
    if maxLoc[0] != 0 and maxLoc[1] != 0 :
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (255,0,0)
        fontScale = 1
        thickness = 2
        mask = cv2.putText(mask , str(maxLoc[0]) + "," + str(maxLoc[1]), maxLoc,font,fontScale,color,thickness,cv2.LINE_AA )

        print(maxLoc)
        cv2.circle(mask , maxLoc , 20 , (255,255,255), 2 , cv2.LINE_AA)

        #conect with image 
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img , "*", maxLoc , font , 0.5, (55,144,144), 1)
        #img = cv2.putText(img , str(maxLoc[0]) + "," + str(maxLoc[1]), maxLoc,font,fontScale,color,thickness,cv2.LINE_AA )
        print(maxLoc)
        cv2.circle(mask , maxLoc , 20 , (255,255,255), 2 , cv2.LINE_AA)
        
    cv2.imshow('fram', mask)
    cv2.imshow('images', img)

    k=cv2.waitKey(1)
    if k == ord ('q'):
        break


cap.release()
cv2.destroyAllWindows()
    

    