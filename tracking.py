import cv2

def event_cap(event, x,y,flags,params):
    #if event == cv2.EVENT_LBUTTONDOWN:
        print("X:" , x , "Y: " ,y)


if __name__ == "__main__":
    img = cv2.imread("picturs/window.jpg")
    img = cv2.resize(img,(500,500))
    cv2.imshow("image", img)



    cv2.setMouseCallback("image", event_cap)
cv2.waitKey(0)
cv2.destroyAllWindows()
