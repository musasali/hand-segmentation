from datetime import time
import timeit

import cv2
import os
import numpy as np


# Display ROI binary mode
def binaryMask(frame, x0, y0, width, height):
    # Display box
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))
    # Extract ROI pixels
    roi = frame[y0 : y0 + height, x0 : x0 + width]
    #
    # Gaussian filtering proces
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Gaussian blur is essentially a low-pass filter, each pixel of the output image is a weighted sum
    # of the pixels surrounding the corresponding pixel on the original image, and Larger the size of the matrix
    # Gaussian, the larger the standard deviation, the greater the degree of the image blur treated
    blur = cv2.GaussianBlur(
        gray, (5, 5), 2
    )  # Gaussian blur, blur matrix and given Gaussian standard deviation

    # When having different brightness of different portions of the same image. In this case, we need to use an
    # adaptive threshold
    # Parameters: src refers to the original image, the original image should be grayscale. x :
    # Refers to the new pixel value that should be given when the pixel value is higher (sometimes less than) the
    # threshold adaptive_method refers to: CV_ADAPTIVE_THRESH_MEAN_C or CV_ADAPTIVE_THRESH_GAUSSIAN_C block_size
    # refers to the pixel neighborhood size used to calculate the threshold : 3, 5, 7, ... param1 refers to the
    # parameters related to the method #
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ret, res = cv2.threshold(
        th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )  # ret or bool type

    " Here you can insert code to call the network "
    # Binarization process
    kernel = np.ones((3, 3), np.uint8)  # Set convolution kernel
    erosion = cv2.erode(
        res, kernel
    )  # etching operation opening operation: the expansion after the first etching,
    # removing isolated dots, glitch
    cv2.imshow("erosion", erosion)
    dilation = cv2.dilate(
        erosion, kernel
    )  # dilation and closing operation: After the first etching expanded,
    # filled pores, small cracks bridging
    cv2.imshow("dilation", dilation)
    #  Contour extraction
    binaryimg = cv2.Canny(res, 50, 200)  # Binarization, canny detection
    h = cv2.findContours(
        binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )  # seek profile
    contours = h[0]  # extracted contour
    ret = np.ones(res.shape, np.uint8)  # Create a black curtain
    cv2.drawContours(ret, contours, -1, (255, 255, 255), 1)  # Draw white outlines
    cv2.imshow("ret", ret)

    # Saving gesture
    if saveImg == True and binaryMode == True:
        saveROI(res)
    elif saveImg == True and binaryMode == False:
        saveROI(roi)
    return res


# Save ROI image
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter > numofsamples:
        # Restored to its original value, in order to continue recording the back gesture
        saveImg = False
        gesturename = ""
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter)  # to record a gesture of naming
    print("Saving img: ", name)
    cv2.imwrite(path + name + ".png", img)  # write files
    time.sleep(0.05)


# Set some commonly used parameters
# Display font size initial position, etc.
font = cv2.FONT_HERSHEY_SIMPLEX  # normal size sans serif
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI box display position
x0 = 300
y0 = 100
# Recorded gesture image size
width = 300
height = 300
# Number of samples per gestures recorded
numofsamples = 300
counter = 0  # counter, recording how many pictures have been recorded
# Memory address and the name of the original folder
gesturename = ""
path = ""
# Identifier bool type is used to represent some state changing needs
binaryMode = False  # whether the ROI is displayed as and to binary mode
saveImg = False  # whether to save the picture

# Create a video capture objects
cap = cv2.VideoCapture(0)  # 0 is (notebook) built-in camera

while True:
    start = timeit.default_timer()
    # Reading frame
    (
        ret,
        frame,
    ) = (
        cap.read()
    )  # # first parameter is returned bool type, to indicate whether the reading frame,
    # if False description has read the last one. frame is the read frame picture
    # Image Flip (Without this step, the video display and we just symmetrical)
    frame = cv2.flip(
        frame, 2
    )  # the second parameter is greater than 0 : it indicates along the y -axis inverted
    # Display ROI area # call the function
    roi = binaryMask(frame, x0, y0, width, height)

    # Display promp
    cv2.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))  # label font
    cv2.putText(
        frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0, 255, 0)
    )  # labeled Font
    cv2.putText(
        frame, "s-'new gestures(twice)'", (fx, fy + 2 * fh), font, size, (0, 255, 0)
    )  # labeled Font
    cv2.putText(
        frame, "q-'quit'", (fx, fy + 3 * fh), font, size, (0, 255, 0)
    )  # labeled Font

    key = cv2.waitKey(1) & 0xFF  # waiting for keyboard input,
    if key == ord("b"):  # The ROI is displayed as binary pattern
        # binaryMode = not binaryMode
        binaryMode = True
        print("Binary Threshold filter active")
    elif key == ord("r"):  # RGB mode
        binaryMode = False

        if key == ord("i"):  # adjusted ROI box
            y0 = y0 - 5
    elif key == ord("k"):
        y0 = y0 + 5
    elif key == ord("j"):
        x0 = x0 - 5
    elif key == ord("l"):
        x0 = x0 + 5

    if key == ord("q"):
        break

    if key == ord("s"):
        """Record a new gesture (training set)"""
        # saveImg = not saveImg # True
        if gesturename != "":  #
            saveImg = True
        else:
            print("Enter a gesture group name first, by enter press 'n'! ")
            saveImg = False
    elif key == ord("n"):
        # Start recording a new gesture
        # First, enter the folder nam
        gesturename = input("enter the gesture folder name: ")
        os.makedirs(gesturename)

        path = (
            "./" + gesturename + "/"
        )  # address generation folder used to store the recorded gesture

    # Show video frame after treatment
    cv2.imshow("frame", frame)
    if binaryMode:
        cv2.imshow("ROI", roi)
    else:
        cv2.imshow("ROI", frame[y0 : y0 + height, x0 : x0 + width])

    stop = timeit.default_timer()
    # display the time it took to run the code
    print("Run time:", start - stop)


# Finally, remember to release the catch
cap.release()
cv2.destroyAllWindows()
