import cv2 as cv
import cv2
import cvzone as cvzone
import numpy as np
from detect_ball import appCalibration


def main():
    # Find ball color bounds
    # lowBounds, highBounds = appCalibration()
    lowBounds, highBounds = (22, 57, 88), (83, 255, 255)

    orb = cv2.ORB_create(15)

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('windows')


    while True:
        ret, frame = cam.read()

        blurred = cv2.GaussianBlur(frame, (11, 11), 0) # Avoid noises
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lowBounds, highBounds)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        # ballMask = cv2.bitwise_and(blurred, blurred, mask=mask)

        imgContours, contours = cvzone.findContours(frame, mask)

        keypoint, des = orb.detectAndCompute(blurred, mask=mask)
        img_keypoints = cv2.drawKeypoints(blurred, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



        cv2.imshow('Img contours', imgContours)
        cv2.imshow('Camera', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('keyPoints', img_keypoints)


        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


main()