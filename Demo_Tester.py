from tester import *
from configurations import *
fileName = "RESIZED-"
fileType = ".jpg"
start=1
end = 4
min_angle = 45
max_angle = 315
min_value = 0
max_value = 1
units = "kg/m3"

def DetectCircle(img):
    img=sharpenImages(img)
    img=adjust_gamma(img)

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


def main():
    for i in range(start,end+1):
        img = cv2.imread(images+fileName + str(i) + fileType)
        width, height = 300, 300
        img = cv2.resize(img, (width, height))
        showImage(img)
        x, y, r = DetectCircle(img)
        if (x == None and y == None and r == None):
            output = "Could not able to get the center point of the needle"
        else:
            r = width * 0.4
            markLabels(img, x, y, r)
            x1, y1, x2, y2 = FindNeedle2(img, x, y, r)
            if (x1 != None and y1 != None and x2 != None and y2 != None):
                output = float(getOutput(x1, y1, x2, y2, min_angle, max_angle, min_value, max_value))
            else:
                output = "Could not able to get the Needle of the Gauge"
        print("*"*100)
        print("file : ",fileName,str(i),fileType)
        print("Output:",output)
        print("*"*100)


def FindNeedle2(img,x, y, r,printFlag=True):
    img = adjust_gamma(img)

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 100
    maxValue = 500

    th, gray2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)
    showImage(gray2,"threshold",True)

    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=gray2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later
    if(printFlag):
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
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
                #cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 2)
    showImage(img,'FOUNDED-LINES',True)
    if(largest_end==0):
        return  (None,None,None,None)

    img = cv2.imread(test_images+'CALIBRATION.jpg')

    drawLine(x,y,end_x,end_y,img)
    showImage(img,"identified_needle",True)
    return x,y,end_x,end_y

if __name__ == '__main__':
    main()
