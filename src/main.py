import cv2 as cv
import cv2
import cvzone as cvzone
import numpy as np
import subprocess
import time
from detect_ball import appCalibration


def depth_update(z, z_step, contours, area, delta_min):
    if abs(contours[0]["area"] - area) > delta_min:
        z = (z - z_step) if contours[0]["area"] < area else (z + z_step) 
        area = contours[0]["area"]

    return z, area


def main(fifo_file):

    first_frame = True

    # Define variable to estimate depth and rotation
    depth_step = 1
    delta_oscillation_min = 1000

    first_area = float()
    current_area = float()
    z = 0
    prev_x = float()
    prev_y = float()

    fixed_keypoints = None # For rotation tracking


    # Find ball color bounds
    # lowBounds, highBounds = appCalibration()
    lowBounds, highBounds = (22, 57, 88), (83, 255, 255)
    orb = cv2.ORB_create(15)

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cam.read()

    h, w, _ = frame.shape
    center = (w // 2, h // 2)

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
                current_area = first_area
                prev_x = contours[0]["center"][0]
                prev_y = contours[0]["center"][1]

            first_frame = False
        else:
            keypoint, des = orb.detectAndCompute(blurred, mask=mask) 
            img_keypoints = cv2.drawKeypoints(blurred, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv2.imshow('keyPoints', img_keypoints)

            img_fixed_keypoints = cv2.drawKeypoints(blurred, fixed_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow('fixed keyPoints', img_fixed_keypoints)

            if contours:
                data = [0, 0, 0] # translation vector

                if abs(contours[0]["center"][0] - prev_x) > 7:
                    data[0] = contours[0]["center"][0] - center[0]  # Shift the coordinates so that (0, 0) is at the center of the image
                    prev_x = contours[0]["center"][0]

                if abs(contours[0]["center"][1] - prev_y) > 7:
                    data[1] = contours[0]["center"][1] - center[1]
                    prev_y = contours[0]["center"][1]

                if abs(contours[0]["area"] - current_area) > delta_oscillation_min:
                    z = (z - depth_step) if contours[0]["area"] < current_area else (z + depth_step) 

                    data[2] = z
                    current_area = contours[0]["area"]


                data = data[0] / w, data[1] / h , data[2] / 100
                # fifo_file.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "\n")
                # fifo_file.flush()
                if (data != (0, 0, 0)):
                    print(data[1])

        cv2.circle(imgContours, center, 5, (0, 0, 255), -1)

        cv2.imshow('Img contours', imgContours)
        # cv2.imshow('Camera', frame)
        # cv2.imshow('Mask', mask)


        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


FILE_PATH = r"\\wsl$\Ubuntu\home\valentin\m2\geometrie_projective\opengl_scene\fifo_trackball"

# Create fifo to communicates
subprocess.run("wsl rm /home/valentin/m2/geometrie_projective/opengl_scene/fifo_trackball && touch /home/valentin/m2/geometrie_projective/opengl_scene/fifo_trackball")

with open(FILE_PATH, "w") as f:
    wsl_command = "wsl /home/valentin/m2/geometrie_projective/opengl_scene/bin/src_main"
    # subprocess.Popen(wsl_command, shell=True)

    main(f)


