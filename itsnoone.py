# # # # # import cv2
# # # # # import numpy as np 

# # # # # def click_event(event, x,y,flags,param):
# # # # #     if event == cv2.EVENT_LBUTTONDOWN:
# # # # #         print("X:",x) , ("Y:",y)
# # # # #         font = cv2.FONT_HERSHEY_COMPLEX
# # # # #         s = str(x) + " , " + str(y)

# # # # #         cv2.putText(frame , s , (x,y) , font , 0.5, (55,144,144), 1)
# # # # #         cv2.imshow('image',frame)


# # # # #         frame = np.zeros((400,720,3),np.uint8)
# # # # #         cv2.imshow("image",frame)

# # # # # cap = cv2.VideoCapture(0)

# # # # # pts = []
# # # # # while (1):

# # # # #     # Take each frame
# # # # #     ret, frame = cap.read()
# # # # #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # # # #     k = cv2.waitKey(1) 
# # # # #     if k == ord('q'):
# # # # #        break



# # # # # cv2.setMouseCallback('image',click_event )

# # # # # cv2.destroyAllWindows()

# # # #//////////////////////////////////////////////////////////////////////////
# # # # import cv2

# # # # # Load image, convert to grayscale, Otsu's threshold
# # # # image = cv2.imread('picturs/window.jpg')
# # # # image = cv2.resize(image,(600,600))
# # # # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# # # # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # # # # Morphological transformations
# # # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
# # # # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

# # # # # Find contours, obtain bounding rect, and find centroid
# # # # cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # # # for c in cnts:
# # # #     # Get bounding rect
# # # #     x,y,w,h = cv2.boundingRect(c)

# # # #     # Find centroid
# # # #     M = cv2.moments(c)
# # # #     cX = int(M["m10"] / M["m00"])
# # # #     cY = int(M["m01"] / M["m00"])

# # # #     # Draw the contour and center of the shape on the image
# # # #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
# # # #     cv2.circle(image, (cX, cY), 1, (320, 159, 22), 8) 
# # # #     cv2.putText(image, '({}, {})'.format(cX, cY), (x,y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
# # # #     print('({}, {})'.format(cX, cY))

# # # # cv2.imshow('image', image)
# # # # cv2.imshow('close', close)
# # # # cv2.imshow('thresh', thresh)
# # # # cv2.waitKey()
# # # #/////////////////////////////////////////////////////////////////////

# # # import numpy as np
# # # import cv2 as cv
# # # from matplotlib import pyplot as plt

# # # img = cv.imread("picturs/window.jpg")
# # # img = cv.resize(img,(600,600))
# # # X = np.random.randint(25,50,(5,2))
# # # Y = np.random.randint(60,85,(4,2))
# # # Z = np.vstack((X,Y))
# # # # convert to np.float32
# # # Z = np.float32(Z)
# # # # define criteria and apply kmeans()
# # # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# # # ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# # # # Now separate the data, Note the flatten()
# # # A = Z[label.ravel()==0]
# # # B = Z[label.ravel()==1]
# # # # Plot the data
# # # plt.scatter(A[:,0],A[:,1])
# # # plt.scatter(B[:,0],B[:,1],c = 'r')
# # # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# # # plt.xlabel('Height'),plt.ylabel('Weight')
# # # plt.show("image", img)
# # #//////////////////////////////////////////////////////////////////////////////////

# # import turtle
 
# # #This to make turtle object
# # #tess=turtle.Turtle() 
 
# # # self defined function to print coordinate
# # def buttonclick(x,y):
# #     print("You clicked at this coordinate({0},{1})".format(x,y))
 
# #  #onscreen function to send coordinate
# # turtle.onscreenclick(buttonclick,1)
# # #turtle.listen()  # listen to incoming connections
# # turtle.speed(10) # set the speed
# # turtle.done()    # hold the screen
# #/////////////////////////////////////////////////////////////////////////

# #! /usr/bin/env python
# import argparse
# from cv2 import cv
# import cv2
# import sys
# import numpy as np
# import serial

# #ser = serial.Serial('COM16', 115200, timeout=1)


# class LaserTracker(object):

#     def __init__(self, cam_width=640, cam_height=480, hue_min=5, hue_max=6,
#                  sat_min=50, sat_max=100, val_min=250, val_max=256,
#                  display_thresholds=False):
#         """
#         * ``cam_width`` x ``cam_height`` -- This should be the size of the
#         image coming from the camera. Default is 640x480.

#         HSV color space Threshold values for a RED laser pointer are determined
#         by:

#         * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
#         * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
#         * ``val_min``, ``val_max`` -- Min/Max allowed pixel values

#         If the dot from the laser pointer doesn't fall within these values, it
#         will be ignored.

#         * ``display_thresholds`` -- if True, additional windows will display
#           values for threshold image channels.

#         """

#         self.cam_width = cam_width
#         self.cam_height = cam_height
#         self.hue_min = hue_min
#         self.hue_max = hue_max
#         self.sat_min = sat_min
#         self.sat_max = sat_max
#         self.val_min = val_min
#         self.val_max = val_max
#         self.display_thresholds = display_thresholds

#         self.capture = None  # camera capture device
#         self.channels = {
#             'hue': None,
#             'saturation': None,
#             'value': None,
#             'laser': None,
#         }

#     def create_and_position_window(self, name, xpos, ypos):
#         """Creates a named widow placing it on the screen at (xpos, ypos)."""
#         # Create a window
#         cv2.namedWindow(name, cv2.CV_WINDOW_AUTOSIZE)
#         # Resize it to the size of the camera image
#         cv2.resizeWindow(name, self.cam_width, self.cam_height)
#         # Move to (xpos,ypos) on the screen
#         cv2.moveWindow(name, xpos, ypos)

#     def setup_camera_capture(self, device_num=0):
#         """Perform camera setup for the device number (default device = 0).
#         Returns a reference to the camera Capture object.

#         """
#         try:
#             device = int(device_num)
#             sys.stdout.write("Using Camera Device: {0}\n".format(device))
#         except (IndexError, ValueError):
#             # assume we want the 1st device
#             device = 0
#             sys.stderr.write("Invalid Device. Using default device 0\n")

#         # Try to start capturing frames
#         self.capture = cv2.VideoCapture(device)
#         if not self.capture.isOpened():
#             sys.stderr.write("Faled to Open Capture device. Quitting.\n")
#             sys.exit(1)

#         # set the wanted image size from the camera
#         self.capture.set(
#             cv.CV_CAP_PROP_FRAME_WIDTH,
#             self.cam_width
#         )
#         self.capture.set(
#             cv.CV_CAP_PROP_FRAME_HEIGHT,
#             self.cam_height
#         )
#         return self.capture

#     def handle_quit(self, delay=10):
#         """Quit the program if the user presses "Esc" or "q"."""
#         key = cv2.waitKey(delay)
#         c = chr(key & 255)
#         if c in ['q', 'Q', chr(27)]:
#             sys.exit(0)

#     def detect(self, frame):
#         hsv_img = cv2.cvtColor(frame, cv.CV_BGR2HSV)

#         LASER_MIN = np.array([0, 0, 230],np.uint8)
#         LASER_MAX = np.array([8, 115, 255],np.uint8)

#         frame_threshed = cv2.inRange(hsv_img, LASER_MIN, LASER_MAX)

#         #cv.InRangeS(hsv_img,cv.Scalar(5, 50, 50),cv.Scalar(15, 255, 255),frame_threshed)    # Select a range of yellow color
#         src = cv.fromarray(frame_threshed)
#         #rect = cv.BoundingRect(frame_threshed, update=0)

#         leftmost=0
#         rightmost=0
#         topmost=0
#         bottommost=0
#         temp=0
#         laserx = 0
#         lasery = 0
#         for i in range(src.width):
#             col=cv.GetCol(src,i)
#             if cv.Sum(col)[0]!=0.0:
#                 rightmost=i
#                 if temp==0:
#                     leftmost=i
#                     temp=1
#         for i in range(src.height):
#             row=cv.GetRow(src,i)
#             if cv.Sum(row)[0]!=0.0:
#                 bottommost=i
#                 if temp==1:
#                     topmost=i
#                     temp=2

#         laserx=cv.Round((rightmost+leftmost)/2)
#         lasery=cv.Round((bottommost+topmost)/2)
#         #return (leftmost,rightmost,topmost,bottommost)


#         return laserx, lasery

#     def display(self, frame):
#         """Display the combined image and (optionally) all other image channels
#         NOTE: default color space in OpenCV is BGR.
#         """
#         cv2.imshow('RGB_VideoFrame', frame)
#         #cv2.imshow('LaserPointer', self.channels['laser'])
#         #if self.display_thresholds:
#          #   cv2.imshow('Thresholded_HSV_Image', img)
#           #  cv2.imshow('Hue', self.channels['hue'])
#            # cv2.imshow('Saturation', self.channels['saturation'])
#             #cv2.imshow('Value', self.channels['value'])

#     def run(self):
#         sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

#         # create output windows
#         #self.create_and_position_window('LaserPointer', 0, 0)
#         self.create_and_position_window('RGB_VideoFrame',
#             10 + self.cam_width, 0)
#         if self.display_thresholds:
#             self.create_and_position_window('Thresholded_HSV_Image', 10, 10)
#             self.create_and_position_window('Hue', 20, 20)
#             self.create_and_position_window('Saturation', 30, 30)
#             self.create_and_position_window('Value', 40, 40)

#         # Set up the camer captures
#         self.setup_camera_capture()

#         while True:
#             # 1. capture the current image
#             success, frame = self.capture.read()
#             if not success:
#                 # no image captured... end the processing
#                 sys.stderr.write("Could not read camera frame. Quitting\n")
#                 sys.exit(1)

#             (laserx, lasery) = self.detect(frame)
#             sys.stdout.write("(" + str(laserx) + "," + str(lasery) + ")" + "\n")
#             #ser.write(str(laserx) + "," + str(lasery) + ",")
#             self.display(frame)


#             self.handle_quit()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Run the Laser Tracker')
#     parser.add_argument('-W', '--width',
#         default=640,
#         type=int,
#         help='Camera Width'
#     )
#     parser.add_argument('-H', '--height',
#         default='480',
#         type=int,
#         help='Camera Height'
#     )
#     parser.add_argument('-u', '--huemin',
#         default=5,
#         type=int,
#         help='Hue Minimum Threshold'
#     )
#     parser.add_argument('-U', '--huemax',
#         default=6,
#         type=int,
#         help='Hue Maximum Threshold'
#     )
#     parser.add_argument('-s', '--satmin',
#         default=50,
#         type=int,
#         help='Saturation Minimum Threshold'
#     )
#     parser.add_argument('-S', '--satmax',
#         default=100,
#         type=int,
#         help='Saturation Minimum Threshold'
#     )
#     parser.add_argument('-v', '--valmin',
#         default=250,
#         type=int,
#         help='Value Minimum Threshold'
#     )
#     parser.add_argument('-V', '--valmax',
#         default=256,
#         type=int,
#         help='Value Minimum Threshold'
#     )
#     parser.add_argument('-d', '--display',
#         action='store_true',
#         help='Display Threshold Windows'
#     )
#     params = parser.parse_args()

#     tracker = LaserTracker(
#         cam_width=params.width,
#         cam_height=params.height,
#         hue_min=params.huemin,
#         hue_max=params.huemax,
#         sat_min=params.satmin,
#         sat_max=params.satmax,
#         val_min=params.valmin,
#         val_max=params.valmax,
#         display_thresholds=params.display
#     )
#     tracker.run()
#//////////////////////////////////////////////////////////
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)

# #pts = []


# while True:

#     # Take each frame
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     def captur_Event(event,x,y,flags,params):
      
#       if event == cv2.EVENT_LBUTTONDOWN:
#         #print(f"({x},{y})")
#         print("X:",x , ",", "Y:",y)

#     if __name__=="__main__":
#         ret, frame = cap.read()
#         lower_red = np.array([0, 0, 255])
#         upper_red = np.array([255, 255, 255])
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
#         cv2.circle(mask, maxLoc, 20, (255, 255, 255), 2, cv2.LINE_AA)
#         print(maxLoc,"x:", "y:")
    
#     #cv2.imshow('Track Laser', mask)
#      #img =cv2.imread("picturs/window.jpg")
#      #img =cv2.resize(img,(600,600))
#     cv2.imshow("Track Laser",mask)
     

     
#     cv2.setMouseCallback("Track Laser",captur_Event)

    

#     #if cv2.waitKey(1) & 0xFF == ord('q'):
#     k=cv2.waitKey(1)
#     if k == ord ('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
img = cv2.imread("picturs/window.jpg")
img = cv2.resize(img,(600,600))
pts = []


while True:

    # Take each frame
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    x1 = 300
    y1 = 300
    x2 = 350
    y2 = 341
    kill = 0
    no_kill = 0
    lower_red = np.array([0, 0, 255])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

    if maxLoc[0] != 0  and maxLoc[1] != 0 :
     font = cv2.FONT_HERSHEY_SIMPLEX # font
     
     fontScale = 1 # fontScale
     color = (255, 0, 0)
     thickness = 2 # Line thickness of 2 px
     mask = cv2.putText(mask, str(maxLoc[0]) + "," + str(maxLoc[1]), maxLoc , font,fontScale, color, thickness, cv2.LINE_AA)
     print(maxLoc)
     cv2.circle(mask, maxLoc, 20, (255, 255, 255), 2, cv2.LINE_AA)

     #to image coordinate
     if maxLoc[1] in range(x1 , x2 ) and maxLoc[0] in range(y1,y2):
        print('Game Over')
     
        font = cv2.FONT_HERSHEY_SIMPLEX # font
        fontScale = 1 # fontScale
        color = (255, 0, 0)
        thickness = 2 # Line thickness of 2 px
        img = cv2.putText(img ,  "*", maxLoc , font,fontScale, color, thickness, cv2.LINE_AA)
        print(maxLoc)
        cv2.circle(mask, maxLoc, 1, (255, 255, 255), 2, cv2.LINE_AA)

    

    cv2.imshow('Track Laser', mask)
    cv2.imshow("image",img)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    k=cv2.waitKey(1)
    if k == ord ('q'):
        break

cap.release()
cv2.destroyAllWindows()