
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

#######   MAIN FUNCTIONS IN THE API #####################
#Step 1
def ProcessImage(image):
    image=sharpenImages(image)
    showImage(image)
    image=adjust_gamma(image,2)
    showImage(image)
    return image

#Step 2

def getCircleAndCustomize(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50,0,30)
    try:
        a, b, c = circles.shape
    except:
        return (None,None,None)
    x,y,r = averageCircle(circles, b) #take average of identified circles for better prediction
    drawCircle(image, x, y, r)
    r+=int(width*0.3)  #increasing the redius by 30 percent
    markLabels(image,x,y,r)
    drawCircle(image,x,y,r)
    return x,y ,r

#STEP 3

def FindNeedle(img,x, y, r):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 100
    maxValue = 500
    th, gray2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=gray2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    print("Number Of Lines Found :", len(lines))
    diff1LowerBound = 0
    diff1UpperBound = 0.30
    diff2LowerBound = 0.5
    diff2UpperBound = 1
    largest_end=0
    end_x=0
    end_y=0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = distance2Points(x, y, x1, y1)  # x, y is center of circle
            diff2 = distance2Points(x, y, x2, y2)
            if (diff1 > diff2):
                diff1,diff2=diff2,diff1
                x1,x2=x2,x1
                y1,y2=y2,y1
            if (((diff1 < diff1UpperBound * r) and (diff1 >=diff1LowerBound * r) and (
                    diff2 <= diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                if(diff2>largest_end):
                    end_x=x2
                    end_y=y2
                    largest_end=diff2
                cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("FOUNDED-LINES.jpg", img)
    if(largest_end==0):
        return  (None,None,None,None)
    img = cv2.imread("CALIBRATION.jpeg")
    drawLine(x,y,end_x,end_y,img,"GAUGE-NEEDLE-IDENTIFIED")
    return x,y,end_x,end_y

#STEP 4
def getOutput(x,y,x2,y2,min_angle,max_angle,min_value,max_value):
    x_angle=x2-x
    y_angle=y-y2
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res
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

def drawLine(x1,y1,x2,y2,image,outName):
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(outName+".jpg", image)
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
    cv2.imwrite("CALIBRATION.jpeg",image)

def resize(image):
    height, width = image.shape[:2]
    dim = (width, height)
    print("Original dimension :height =" + str(height) + "\t -- Width =" + str(width))
    scale = 1  # for adjusting the image size just for testing
    if (height > 720):
        scale = .50
    if (height > 1080):
        scale = 0.40
    if (height > 2000):
        scale = 0.38
    if (height > 2200):
        scale = 0.30
    height = int(height * scale)
    width = int(width * scale)
    dim = (width, height)
    if (scale != 1):
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print("AFTER Resize: height =" + str(height) + "\t  Width =" + str(width))
    return image

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
            cv2.imwrite(name+".jpg",img)
        k = cv2.waitKey(33)
        if k == 27:
            break
        elif k == -1:
            continue