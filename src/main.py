import cv2 as cv
import cv2
import cvzone as cvzone
import numpy as np
from detect_ball import appCalibration


def main():

    first_frame = True

    # Define variable to estimate depth and rotation
    depth_step = 1
    first_area = float()
    delta_oscillation_max = 10000
    z = 0
    fixed_keypoints = None

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

        imgContours, contours = cvzone.findContours(frame, mask)


        # Process for the first frame
        if first_frame:
            fixed_keypoints, des = orb.detectAndCompute(blurred, mask=mask)        
            if contours:
                first_area = contours[0]["area"]

            first_frame = False
        else:
            keypoint, des = orb.detectAndCompute(blurred, mask=mask) 
            img_keypoints = cv2.drawKeypoints(blurred, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv2.imshow('keyPoints', img_keypoints)

            img_fixed_keypoints = cv2.drawKeypoints(blurred, fixed_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('fixed keyPoints', img_fixed_keypoints)

            if contours and abs(contours[0]["area"] - first_area) > delta_oscillation_max:
                
                if contours[0]["area"] < first_area:
                    z -= depth_step
                else:
                    z += depth_step
                area = contours[0]["area"]
                print(f"first area {first_area} area = {area} z = {z}")



        # Get position and area from contours
        if contours:
            data = contours[0]["center"][0], contours[0]["center"][1], contours[0]["area"]


        cv2.imshow('Img contours', imgContours)
        cv2.imshow('Camera', frame)
        cv2.imshow('Mask', mask)


        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


main()