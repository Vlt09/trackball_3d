import cv2 as cv
import cv2
import cvzone as cvzone
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import math

from detect_ball import appCalibration


def depth_update(z, z_step, contours, area, delta_min):
    if abs(contours[0]["area"] - area) > delta_min:
        z = (z - z_step) if contours[0]["area"] < area else (z + z_step) 
        area = contours[0]["area"]

    return z, area

"""
This function takes and image and keypoints from previous
frame to estimate rotation. Rotation estimation is based on  
distance of 2 similar keypoints from 2 successive frame.  
"""
def rotation_process(prev_frame, img_input, img_mask, prev_keypoints, desc_1, orb, h, w):
    new_kp, desc_2 = orb.detectAndCompute(img_input, mask=img_mask)
    nb_kp = 5
    sum_angle = 0

    # Initialize BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(desc_1, desc_2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    for i in range(1):
        kp_img1 = prev_keypoints[matches[i].queryIdx]
        kp_img2 = new_kp[matches[i].trainIdx]

        x1, y1 = kp_img1.pt
        x2, y2 = kp_img2.pt

        # x1 = (w - x1) / w
        # x2 = (w - x2) / w

        # y1 = (h - y1) / h
        # y2 = (h - y2) / h

        # sum_angle += math.atan2(y2 - y1, x2 - x1)
        sum_angle += math.atan((x2 - x1) / w)
        # print(f"for i = {i} angle = {math.degrees(math.atan2(y2 - y1, x2 - x1))} pt1 {(x1, y1)} pt2 {(x2, y2)}")

    # Draw first 10 matches.
    img3 = cv.drawMatches(prev_frame,prev_keypoints,img_input,new_kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(img3),plt.show()

    # prev_keypoints = new_kp
    return math.degrees(sum_angle / 1)


def main(fifo_file):

    first_frame_flag = True
    first_frame = None

    # Define variable to estimate depth and rotation
    depth_step = 1
    delta_oscillation_min = 1000

    first_area = float()
    current_area = float()
    z = 0
    prev_x = float()
    prev_y = float()

    fixed_keypoints = None # For rotation tracking
    desc = None

    # Find ball color bounds
    # lowBounds, highBounds = appCalibration()
    lowBounds, highBounds = (22, 57, 88), (83, 255, 255)
    orb = cv2.ORB_create()

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
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        imgContours, contours = cvzone.findContours(frame, mask)


        # Process for the first frame
        if first_frame_flag:
            first_frame = blurred
            fixed_keypoints, desc = orb.detectAndCompute(blurred, mask=mask)        
            if contours:
                first_area = contours[0]["area"]
                current_area = first_area
                prev_x = contours[0]["center"][0]
                prev_y = contours[0]["center"][1]

            first_frame_flag = False
        elif contours:
            data = [0, 0, 0] # translation vector

            if abs(contours[0]["center"][0] - prev_x) > 7:
                data[0] = (w - contours[0]["center"][0]) - center[0]  # Shift the coordinates so that (0, 0) is at the center of the image
                prev_x = contours[0]["center"][0]

            if abs(contours[0]["center"][1] - prev_y) > 7:
                data[1] = (h - contours[0]["center"][1]) - center[1]
                prev_y = contours[0]["center"][1]

            if abs(contours[0]["area"] - current_area) > delta_oscillation_min:
                z = (z - depth_step) if contours[0]["area"] < current_area else (z + depth_step) 

                data[2] = z
                current_area = contours[0]["area"]


            data = (data[0]) / w, data[1] / h , data[2] / 100
            # fifo_file.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "\n")
            # fifo_file.flush()
            # if (data[1] != 0):
            #     print(data[1])
            print(rotation_process(first_frame, blurred, mask, fixed_keypoints, desc, orb, h, w))

            # x, y, w, h = contours[0]["bbox"]

            # # Extraire uniquement la zone de la bounding box
            # ball_only_img = np.zeros_like(frame) 
            # ball_only_img[y:y+h, x:x+w] = frame[y:y+h, x:x+w]  # Copier seulement la région de la bbox
            # cv2.imshow("Ball Extracted", ball_only_img)


        cv2.circle(imgContours, center, 5, (0, 0, 255), -1)

        cv2.imshow('Img contours', imgContours)
        cv2.imshow('Camera', blurred)
        cv2.imshow('Mask', mask)


        if cv2.waitKey(1) == ord('q'):
            break

        # if cv2.waitKey(1) == ord('p'):
        #     cv2.waitKey(120)


    cam.release()
    cv2.destroyAllWindows()


FILE_PATH = r"\\wsl$\Ubuntu\home\valentin\m2\geometrie_projective\opengl_scene\fifo_trackball"

# Create fifo to communicates
# subprocess.run("wsl rm /home/valentin/m2/geometrie_projective/opengl_scene/fifo_trackball && touch /home/valentin/m2/geometrie_projective/opengl_scene/fifo_trackball")

with open(FILE_PATH, "w") as f:
    wsl_command = "wsl /home/valentin/m2/geometrie_projective/opengl_scene/bin/src_main"
    # subprocess.Popen(wsl_command, shell=True)

    main(f)


