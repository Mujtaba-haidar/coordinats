import cv2
import matplotlib.pyplot as plt


image= cv2.imread('picturs/window.jpg')

original_image= image

gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

edges= cv2.Canny(gray, 50,200)

contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

sorted_contours= sorted(contours, key=cv2.contourArea, reverse= False)

smallest_item= sorted_contours[0]

largest_item= sorted_contours[1]

#largest item
M= cv2.moments(largest_item)

x,y,w,h= cv2.boundingRect(largest_item)

xcoordinate1= x 

xcoordinate2= x + w

xcoordinate_center= int(M['m10']/M['m00'])


# print("Larger Box")

# print("x coordinate 1: ", str(xcoordinate1))

# print("x coordinate 2: ", str(xcoordinate2))

# print("x center coordinate ", str(xcoordinate_center))

# print("")


ycoordinate1= y 

ycoordinate2= y + h

ycoordinate_center= int(M['m01']/M['m00'])



# print("y coordinate 1: ", str(ycoordinate1))

# print("y coordinate 2: ", str(ycoordinate2))

# print("y center coordinate ", str(ycoordinate_center))


# print("")

#largest item
M2= cv2.moments(smallest_item)

x2,y2,w2,h2= cv2.boundingRect(smallest_item)

x2coordinate1= x2 

x2coordinate2= x2 + w2

x2coordinate_center= int(M2['m10']/M2['m00'])


# print ("Smaller Box")

# print("x coordinate 1: ", str(x2coordinate1))

# print("x coordinate 2: ", str(x2coordinate2))

# print("x center coordinate ", str(x2coordinate_center))

# print ("")


y2coordinate1= y2 

y2coordinate2= y2 + h2

y2coordinate_center= int(M2['m01']/M2['m00'])


# print("y coordinate 1: ", str(y2coordinate1))

# print("y coordinate 2: ", str(y2coordinate2))

# print("y center coordinate ", str(y2coordinate_center))


plt.imshow(image)
plt.show()

