from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
# Initialize the Flask application
from FLASK_BACKuP.tester import *


def DetectCircle(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    showImage(output_img,"masked",True)
    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    showImage(output_hsv,"HSV",True)

    # Convert to grayscale.
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    showImage(gray_blurred,"blurred")
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)
    a, b, r = None, None, None
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            showImage(img,"Identified_circle",True)
            break
    return a, b, r


def main():
    min_angle = 45
    max_angle = 315
    min_value = 0
    max_value = 10
    units = "kg/m3"

    print(min_value, min_angle, max_angle, max_value, units)
    # do some fancy processing here....

    output = "Could not able to process"

    img = cv2.imread("RESIZED.jpg")
    width,height=300,300
    img=cv2.resize(img,(width,height))
    showImage(img)

    x, y, r = DetectCircle(img)
    if (x == None and y == None and r == None):
        output = "Could not able to get the center point of the needle"
    else:
        r=width*0.4
        markLabels(img,x,y,r)
        x1, y1, x2, y2 = FindNeedle(img, x, y, r)
        if (x1 != None and y1 != None and x2 != None and y2 != None):
            output = float(getOutput(x1, y1, x2, y2, min_angle, max_angle, min_value, max_value))
            print("Final output :" + str(output))
        else:
            output = "Could not able to get the Needle of the Gauge"

    print(output)


if __name__ == '__main__':
    main()

