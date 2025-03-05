import cv2 as cv
import cv2


def setLowBounds(bounds: list, value: int):
    bounds[0] = value

def setHighBounds(bounds: list, value: int):
    bounds[1] = value

def resetBounds(bounds: list):
    bounds = [0, 255]

"""
This function creates trackbars allowing the user to find the values 
for which only the ball is visible on the binary image.
"""
def setupCalibrationParameters(windowsName: str, redBounds: list, blueBounds: list, greenBounds: list):

    cv2.createTrackbar('Low red', windowsName, 0, 255, lambda x: setLowBounds(redBounds, x))
    cv2.createTrackbar('High Red', windowsName, 0, 255, lambda x: setHighBounds(redBounds, x))
    cv2.createTrackbar('Low green', windowsName, 0, 255, lambda x: setLowBounds(greenBounds, x))
    cv2.createTrackbar('High green', windowsName, 0, 255, lambda x: setHighBounds(greenBounds, x))
    cv2.createTrackbar('Low blue', windowsName, 0, 255, lambda x: setLowBounds(blueBounds, x))
    cv2.createTrackbar('High blue', windowsName, 0, 255, lambda x: setHighBounds(blueBounds, x))

"""
This function is executed first and is used to calibrate the application. 
It is the step where the user must indicate where the ball is to be tracked and sets the threshold 
so that only the white ball is seen in grayscale.
"""
def appCalibration():

    redBounds = [0, 255]
    greenBounds = [0, 255]
    blueBounds = [0, 255]
    windowsName = 'Parameter setting'

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(windowsName)

    setupCalibrationParameters(windowsName, redBounds, blueBounds, greenBounds)


    while True:
        ret, frame = cam.read()
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        frame2 = cv.inRange(blurred, (redBounds[0], greenBounds[0], blueBounds[0]), (redBounds[1], greenBounds[1], blueBounds[1]))

        cv2.imshow('Camera 2', blurred)
        cv2.imshow('Camera', frame2)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    return (redBounds[0], greenBounds[0], blueBounds[0]), (redBounds[1], greenBounds[1], blueBounds[1])
