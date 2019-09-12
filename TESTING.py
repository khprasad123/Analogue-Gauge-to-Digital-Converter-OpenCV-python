
import os
import cv2
import numpy as np

def getCurrentPath():
    currentFile = __file__
    realPath = os.path.realpath(currentFile)
    dirPath = os.path.dirname(realPath)
    return dirPath
def setStaticUserRealGaugeDetails():
    min_angle = 45 # input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    max_angle = 315 # input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    min_value = 0 #input('Min value: ') #usually zero
    max_value = 10 #input('Max value: ') #maximum reading of the gauge
    units = 'kg/cm3' #input('Enter units: ')
    return float(min_angle),float(max_angle),float(min_value),float(max_value),units
def resize(image):
    height, width = image.shape[:2]
    dim=(width,height)
    print("Original dimention -- height =" + str(height) + "\t -- Width =" + str(width))
    scale = 1  # for adjusting the image size just for testing
    if(height>720):
        scale=.50
    if(height>1080):
        scale=0.40
    if(height>2000):
        scale=0.30
    height = int(height * scale)
    width = int(width * scale)
    dim = (width, height)
    if(scale!=1):
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print("AFTER Resize -- height =" + str(height) + "\t -- Width =" + str(width))
    return image
def showImage(image):
    cv2.imshow("Chumma",image)
    cv2.waitKey()
    cv2.destroyAllWindows()
def sharpenImages(image):
    # Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!
    # Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    # Subtract the two:
    kernel = kernel - boxFilter
    # Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.
    custom = cv2.filter2D(image, -1, kernel)
    return custom
def findGuageValue(x,y,x2,y2):
    min_angle, max_angle, min_value, max_value, units = setStaticUserRealGaugeDetails()
    #getting Slope
    x_angle=x2-x
    y_angle=y-y2
    # take the arc tan of y/x to find the angle
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
    print(final_angle)
    angle_Range=max_angle-min_angle
    value_Range=max_value-min_value
    output=(((final_angle-min_angle)*value_Range)/angle_Range)+min_value
    return output
def drawLine(x1,y1,x2,y2,image,outName):
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(outName+".jpg", image)
    return image
def getImage():
    img="IMAGE-EDITED-3.jpg"
    image=getCurrentPath()+"/images/"+img
    image=cv2.imread(image)
    image=sharpenImages(image)
    image=adjust_gamma(image,2)
    return resize(image)
def distance2Points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def markLabels(image,x,y,r):
    min_angle, max_angle, min_value, max_value, units=setStaticUserRealGaugeDetails()
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
    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(image, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(image, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
    cv2.imwrite("caliberation-stage.jpeg",image)
def drawCircle(image,x,y,r):
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return image
def averageCircle(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r
    #return the avg center and avg radius of the circle

def getCircleAndCustomize(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    showImage(gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, 0,int(width*0.1))
    a, b, c = circles.shape
    x,y,r = averageCircle(circles, b)
    showImage(drawCircle(image, x, y, r))
    r+=int(width*0.3)  #increasing the redius by 30 percent
    markLabels(image,x,y,r)
    showImage(drawCircle(image,x,y,r))
    return x,y ,r

def getOutputValue(img,x, y, r):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 100
    maxValue = 500
    th, gray2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    showImage(gray2)
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=gray2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    print("Number Of Lines Found :",len(lines))
    diff1LowerBound = 0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.30
    diff2LowerBound = 0.5  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1
    largest_end=0
    end_x=0
    end_y=0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = distance2Points(x, y, x1, y1)  # x, y is center of circle
            diff2 = distance2Points(x, y, x2, y2)
            # x, y is center of circle
            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                diff1,diff2=diff2,diff1
                x1,x2=x2,x1
                y1,y2=y2,y1
            # check if line is within an acceptable range
            if (((diff1 < diff1UpperBound * r) and (diff1 >=diff1LowerBound * r) and (
                    diff2 <= diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                if(diff2>largest_end):
                    end_x=x2
                    end_y=y2
                    largest_end=diff2
                cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("FOUNDED-LINES.jpg", img)
    showImage(img)
    if(largest_end==0):
        return  "No Gauge Needle Detected - Lighting or Image Issues " #that means the image having no lines inside so the needle is not detected this is and EXCEPTION
    img = cv2.imread("caliberation-stage.jpeg")
    showImage(drawLine(x,y,end_x,end_y,img,"Needle-Identified"))
    return (findGuageValue(x,y,end_x,end_y))

def main():
    image= getImage()
    x,y,r = getCircleAndCustomize(image)
    newValue = getOutputValue(image,x,y,r)
    print(newValue)
def getCurrentPath():
    currentFile = __file__
    realPath = os.path.realpath(currentFile)
    dirPath = os.path.dirname(realPath)
    return dirPath
def setStaticUserRealGaugeDetails():
    min_angle = 45 # input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    max_angle = 315 # input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    min_value = 0 #input('Min value: ') #usually zero
    max_value = 10 #input('Max value: ') #maximum reading of the gauge
    units = 'kg/cm3' #input('Enter units: ')
    return float(min_angle),float(max_angle),float(min_value),float(max_value),units
def resize(image):
    height, width = image.shape[:2]
    dim=(width,height)
    print("Original dimention -- height =" + str(height) + "\t -- Width =" + str(width))
    scale = 1  # for adjusting the image size just for testing
    if(height>720):
        scale=.50
    if(height>1080):
        scale=0.40
    if(height>2000):
        scale=0.30
    height = int(height * scale)
    width = int(width * scale)
    dim = (width, height)
    if(scale!=1):
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print("AFTER Resize -- height =" + str(height) + "\t -- Width =" + str(width))
    return image
def showImage(image):
    cv2.imshow("Chumma",image)
    cv2.waitKey()
    cv2.destroyAllWindows()
def sharpenImages(image):
    # Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0  # Identity, times two!
    # Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    # Subtract the two:
    kernel = kernel - boxFilter
    # Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.
    custom = cv2.filter2D(image, -1, kernel)
    return custom
def findGuageValue(x,y,x2,y2):
    min_angle, max_angle, min_value, max_value, units = setStaticUserRealGaugeDetails()
    #getting Slope
    x_angle=x2-x
    y_angle=y-y2
    # take the arc tan of y/x to find the angle
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
    print(final_angle)
    angle_Range=max_angle-min_angle
    value_Range=max_value-min_value
    output=(((final_angle-min_angle)*value_Range)/angle_Range)+min_value
    return output
def drawLine(x1,y1,x2,y2,image,outName):
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(outName+".jpg", image)
    return image
def getImage():
    img="IMAGE-EDITED.jpg"
    image=getCurrentPath()+"/images/"+img
    image=cv2.imread(image)
    image=sharpenImages(image)
    image=adjust_gamma(image,2)
    return resize(image)
def distance2Points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def markLabels(image,x,y,r):
    min_angle, max_angle, min_value, max_value, units=setStaticUserRealGaugeDetails()
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
    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(image, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(image, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
    cv2.imwrite("caliberation-stage.jpeg",image)
def drawCircle(image,x,y,r):
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return image
def averageCircle(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r
    #return the avg center and avg radius of the circle

def getCircleAndCustomize(image):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    showImage(gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, 0,int(width*0.10))
    a, b, c = circles.shape
    x,y,r = averageCircle(circles, b)
    showImage(drawCircle(image, x, y, r))
    r+=int(width*0.3)  #increasing the redius by 30 percent
    markLabels(image,x,y,r)
    showImage(drawCircle(image,x,y,r))
    return x,y ,r

def getOutputValue(img,x, y, r):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 100
    maxValue = 500
    th, gray2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    showImage(gray2)
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=gray2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    print("Number Of Lines Found :",len(lines))
    diff1LowerBound = 0  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.30
    diff2LowerBound = 0.5  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1
    largest_end=0
    end_x=0
    end_y=0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = distance2Points(x, y, x1, y1)  # x, y is center of circle
            diff2 = distance2Points(x, y, x2, y2)
            # x, y is center of circle
            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                diff1,diff2=diff2,diff1
                x1,x2=x2,x1
                y1,y2=y2,y1
            # check if line is within an acceptable range
            if (((diff1 < diff1UpperBound * r) and (diff1 >=diff1LowerBound * r) and (
                    diff2 <= diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                if(diff2>largest_end):
                    end_x=x2
                    end_y=y2
                    largest_end=diff2
                cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("FOUNDED-LINES.jpg", img)
    showImage(img)
    if(largest_end==0):
        return  "No Gauge Needle Detected - Lighting or Image Issues " #that means the image having no lines inside so the needle is not detected this is and EXCEPTION
    img = cv2.imread("caliberation-stage.jpeg")
    showImage(drawLine(x,y,end_x,end_y,img,"Needle-Identified"))
    return (findGuageValue(x,y,end_x,end_y))

def main():
    image= getImage()
    x,y,r = getCircleAndCustomize(image)
    newValue = getOutputValue(image,x,y,r)
    print(newValue)
if __name__=='__main__':
    main()