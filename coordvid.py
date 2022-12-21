import cv2
import numpy as np 

rect = (0,0,0,0)
startPoint = False
endPoint = False
selected_ROI = False

def on_mouse(event,x,y,flags,params):

    global rect,startPoint,endPoint, selected_ROI

    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)
            selected_ROI = False

        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True

        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True
            selected_ROI = True

video = cv2.VideoCapture(0)

while video.isOpened():
    # Read frame

    ret, frame = video.read()

    cv2.namedWindow('original')
    cv2.setMouseCallback('original', on_mouse)

    if selected_ROI:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)

  
    region = frame[rect[0]:rect[1], rect[2]:rect[3]]
    cv2.imshow('original', frame)
    
    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
    
#     cv2.imshow("video",frame)
#     k= cv2.waitKey(1)
#     if k == ord("q"):
#         break

# #cv2.release()
# cv2.destroyAllWindows()