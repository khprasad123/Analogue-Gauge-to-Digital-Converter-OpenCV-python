
import cv2
import numpy as np
from configurations import *

#######   MAIN FUNCTIONS IN THE API #####################
#Step 1
def ProcessForCenterDetection(img):
    img = sharpenImages(img)
    img = adjust_gamma(img)

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

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    # Convert to grayscale.
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    showImage(gray)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    return gray_blurred

#Step 2

def DetectCenter(img):
    gray_blurred=img
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=25, minRadius=1, maxRadius=40)
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

#STEP 3

def FindNeedle(img,x, y, r,printFlag=False):
    temp=img
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 100
    maxValue = 500

    th, gray2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    showImage(gray2, "threshold", True)

    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=gray2, rho=3, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,
                            maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    if (printFlag):
        print("Number Of Lines Found :", len(lines))
    diff1LowerBound = 0
    diff1UpperBound = 0.30
    diff2LowerBound = 0.5
    diff2UpperBound = 1
    largest_end = 0
    end_x = 0
    end_y = 0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            diff1 = distance2Points(x, y, x1, y1)  # x, y is center of circle
            diff2 = distance2Points(x, y, x2, y2)
            if (diff1 > diff2):
                diff1, diff2 = diff2, diff1
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if (((diff1 < diff1UpperBound * r) and (diff1 >= diff1LowerBound * r) and (
                    diff2 <= diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                if (diff2 > largest_end):
                    end_x = x2
                    end_y = y2
                    largest_end = diff2
                # cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
    if (largest_end == 0):
        return (None, None, None, None)

    drawLine(x, y, end_x, end_y, temp)
    showImage(temp, "identified_needle", True)
    return x, y, end_x, end_y


#STEP 4
def getOutput(x,y,x2,y2,min_angle,max_angle,min_value,max_value,printFlag=False):
    x_angle=x2-x
    y_angle=y-y2
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    res = np.rad2deg(res)
    final_angle=0
    if x_angle > 0 and y_angle > 0:  # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res
    if(printFlag):
        print("Angle of needle :"+str(final_angle))
    angle_Range=max_angle-min_angle
    value_Range=max_value-min_value
    output=(((final_angle-min_angle)*value_Range)/angle_Range)+min_value
    return output

########################################################################################################
#Below functions are supporting functions for the main functions

def drawCircle(image,x,y,r):
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return image

def drawLine(x1,y1,x2,y2,image):
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def markLabels(image,x,y,r):
    # The Range Marking
    separation = 10 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
    # #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(image, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(image, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
    showImage(image,"CALIBRATION",True)


def sharpenImages(image):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(image, -1, kernel)
    return custom


def distance2Points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def averageCircle(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r


def showImage(img,name="test",save=False):
    while (1):
        cv2.imshow(name, img)
        if(save):
            cv2.imwrite(test_images+name+".jpg",img)
        k = cv2.waitKey()
        if k == 27:
            break
        elif k == -1:
            continue