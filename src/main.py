import cv2 as cv
import cv2
import cvzone as cvzone
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import math

from detect_ball import appCalibration

def calculate_farneback_optical_flow(img1, img2, center):
    """
    Calculate the dense optical flow using Farneback method.
    
    Parameters:
    img1 (numpy.ndarray): The first input image (grayscale).
    img2 (numpy.ndarray): The second input image (grayscale).
    
    Returns:
    flow (numpy.ndarray): A 2D array of flow vectors representing pixel displacements.
    """

    translation_matrix = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])
    img2_aligned = cv2.warpAffine(img2, translation_matrix, (img2.shape[1], img2.shape[0]))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    img2_gray = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY) if len(img2_aligned.shape) == 3 else img2_aligned
    
    # Optical flow with Farneback method
    return cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def visualize_flow(flow):
    """
    Visualize the optical flow as a color-coded image.
    
    Parameters:
    flow (numpy.ndarray): The optical flow result.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image with 3 channels
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    hsv[..., 1] = 255  # Set saturation to maximum
    hsv[..., 0] = angle * 180 / np.pi / 2  
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical Flow', flow_rgb)

def compute_rotation_axis(flow, center, radius):
    """
    Compute the axis of rotation based on the optical flow in the region of the ball.
    
    Parameters:
    flow (numpy.ndarray): The optical flow (deformation vector field).
    center (tuple): The center of the ball (x, y).
    radius (int): The radius of the ball.
    
    Returns:
    axis_of_rotation (numpy.ndarray): The unit vector representing the axis of rotation.
    """
    # Create a mask for the ball region
    mask = np.zeros_like(flow[..., 0], dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    
    masked_flow = flow * mask[..., np.newaxis]  
    
    # Compute the average displacement vector (dx, dy) in the ball region
    average_vector = np.mean(masked_flow, axis=(0, 1))  
    # print("Average DVF vector (dx, dy):", average_vector)
    
    axis_of_rotation = np.array([-average_vector[1], average_vector[0]]) 
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)
    
    return axis_of_rotation

def calculate_rotation_angle(flow, center, radius):
    mask = np.zeros_like(flow[..., 0], dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, thickness=-1)  # Ball mask
    
    masked_flow = flow * mask[..., np.newaxis]  # Element-wise multiplication
    
    angle_displacements = np.arctan2(masked_flow[..., 1], masked_flow[..., 0])  # Compute the angle for each point's flow vector
    
    mean_angle = np.mean(angle_displacements)  # Average angle
    mean_angle_deg = np.degrees(mean_angle)
    
    return mean_angle_deg

def angle_between_points(A, B, C):
    """
    Computes the angle formed at point B by the points A, B, and C.

    :param A: Tuple (x, y) or (x, y, z) representing point A
    :param B: Tuple (x, y) or (x, y, z) representing point B (vertex of the angle)
    :param C: Tuple (x, y) or (x, y, z) representing point C
    :return: Angle in degrees
    """
    BA = np.array(A) - np.array(B)
    BC = np.array(C) - np.array(B)

    dot_product = np.dot(BA, BC)

    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    angle_rad = np.arccos(dot_product / (norm_BA * norm_BC))

    return np.degrees(angle_rad)

def depth_update(z, z_step, contours, area, delta_min):
    if abs(contours[0]["area"] - area) > delta_min:
        z = (z - z_step) if contours[0]["area"] < area else (z + z_step) 
        area = contours[0]["area"]

    return z, area


def rotation_process(prev_frame, img_input, img_mask, prev_keypoints, desc_1, orb, center, angle):
    """
    This function takes and image and keypoints from previous
    frame to estimate rotation. Rotation estimation is based on  
    distance of 2 similar keypoints from 2 successive frame.  
    """
    translation_matrix = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])
    img_aligned = cv2.warpAffine(img_input, translation_matrix, (img_input.shape[1], img_input.shape[0]))
    img_aligned = cv.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    new_kp, desc_2 = orb.detectAndCompute(img_aligned, None)
    axis = None
    sum_angle = 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    try:
        matches = bf.match(desc_1, desc_2)
    except cv2.error as e:
        return [0, 0, 0], 0
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return [0, 0, 0], 0

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    nb_matches = 1

    kp_img1 = prev_keypoints[matches[0].queryIdx]
    kp_img2 = new_kp[matches[0].trainIdx]

    x1, y1 = kp_img1.pt
    x2, y2 = kp_img2.pt

    deltaX = abs(x1 - x2)
    deltaY = abs(y1 - y2)
    dirX = 1 if x2 >= x1 else -1
    dirY = 1 if y2 >= y1 else -1

    if deltaX - deltaY < 0: 
        axis = [0, 0, dirX]
    elif deltaX - deltaY > 0:
        axis = [0, dirY, 0]
    elif deltaX - deltaY < 0.00001:
        axis = [0, dirY, dirX]

    # sum_angle += math.atan2(y2 - y1, x2 - x1)
    sum_angle += angle_between_points((x1, y1), center, (x2, y2))
    # print(f"for i = {i} angle = {math.degrees(math.atan2(y2 - y1, x2 - x1))} pt1 {(x1, y1)} pt2 {(x2, y2)}")

    # Draw first 10 matches.
    #img3 = cv.drawMatches(prev_frame,prev_keypoints,img_aligned,new_kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    # plt.imshow(img3),plt.show()

    sum_angle /= nb_matches

    if abs(angle - sum_angle) > 5 or sum_angle == math.nan:
        return [0, 0, 0], 0

    prev_keypoints = new_kp
    desc_1 = desc_2

    return axis, sum_angle

def calculate_translation_and_depth(contours, prev_x, prev_y, current_area, center, w, h, z, delta_oscillation_min, depth_step):
    data = [0, 0, 0]

    # Avoid micro mouvement between 2 frames
    if abs(contours[0]["center"][0] - prev_x) > 50:
        direction = 1 if contours[0]["center"][0] >= center[0] else -1 

        # Sub by screen width to inverse x decreasing in openGL scene
        data[0] = (w - contours[0]["center"][0]) * direction
        prev_x = contours[0]["center"][0]

    if abs(contours[0]["center"][1] - prev_y) > 50:
        direction = 1 if contours[0]["center"][1] >= center[1] else -1 

        # Sub by screen height to inverse y decreasing in openGL scene
        data[1] = (h - contours[0]["center"][1]) * direction
        prev_y = contours[0]["center"][1]

    if abs(contours[0]["area"] - current_area) > delta_oscillation_min:
        z = (z + depth_step) if contours[0]["area"] < current_area else (z - depth_step) 
        data[2] = z
        current_area = contours[0]["area"]

    data = (data[0]) / w, data[1] / h , data[2] / 100

    return data, prev_x, prev_y, current_area, z

def process_frame(frame, mask, prev_center):
    movement_threshold = 5

    # Find contours in the current frame
    imgContours, contours = cvzone.findContours(frame, mask)

    if contours:
        current_center = contours[0]["center"]

        if prev_center is not None:
            distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))
            
            # Avoid any process if movement is too small
            if distance < movement_threshold:
                return 0, imgContours, contours
            return 1, imgContours, contours
    else:
        return 0, None, None


def main(fifo_file, webcam_flag):

    first_frame_flag = True
    first_frame = None

    # Define variable to estimate depth and rotation
    depth_step = 1
    delta_oscillation_min = 2000

    first_ball_center = float()
    first_area = float()
    current_area = float()
    prev_x = float()
    prev_y = float()
    z = 0
    angle = 0

    fixed_keypoints = None # For rotation tracking
    desc = None

    # Find ball color bounds
    # lowBounds, highBounds = appCalibration()
    lowBounds, highBounds = (22, 57, 88), (83, 255, 255)
    orb = cv2.ORB_create()

    # Open the default camera
    cam = cv2.VideoCapture(0)
    if cam.isOpened() is False:
        print("Error with video")
        return

    # Get the default frame width and height
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


    first_ball_img = None
    ret, frame = cam.read()

    # h, w, _ = frame.shape
    center = (w // 2, h // 2)

    cv2.namedWindow('windows')
    while True:
        ret, frame = cam.read()

        blurred = cv2.GaussianBlur(frame, (11, 11), 0) # Avoid noises
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) if webcam_flag else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if webcam_flag:
            mask = cv2.inRange(hsv, lowBounds, highBounds)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
        else:
            mask = hsv

        # imgContours, contours = cvzone.findContours(blurred, mask)
        flag, imgContours, contours = process_frame(blurred, mask, center)

        # Process for the first frame
        if first_frame_flag and flag == 1:
            first_frame = imgContours
            fixed_keypoints, desc = orb.detectAndCompute(imgContours, mask=mask)        
            if contours:
                first_area = contours[0]["area"]
                first_ball_center = contours[0]["center"]
                current_area = first_area
                prev_x = contours[0]["center"][0]
                prev_y = contours[0]["center"][1]
                first_ball_img = cv2.bitwise_and(blurred, blurred, mask=mask)

            first_frame_flag = False
        elif contours and flag == 1:
            data = calculate_translation_and_depth(contours, prev_x, prev_y, current_area, first_ball_center, w, h, z, delta_oscillation_min, depth_step)
            result = cv2.bitwise_and(blurred, blurred, mask=mask)

            prev_x = data[1]
            prev_y = data[2]
            current_area = data[3]
            z = data[4]

            p1 = np.array(contours[0]["center"])
            p2 = np.array(first_ball_center)
            
            c = (p1[0] - p2[0], p1[1] - p2[1])
            direction, theta = rotation_process(first_ball_img, result, mask, fixed_keypoints, desc, orb, c, angle)
            print(theta)

            fifo_file.write(str(data[0][0]) + "," + str(data[0][1]) + "," + str(data[0][2]) + "," \
                            + str(theta) + "," + str(direction[0]) + "," + str(direction[1]) + "," + str(direction[2]) + "\n")
            fifo_file.flush()

            angle = theta if theta != 0 else angle # keep angle value which is not equal to 0
            first_ball_center = contours[0]["center"]
            first_ball_img = result

            cv2.circle(imgContours, center, 5, (0, 0, 255), -1)

        cv2.imshow('Img contours', imgContours)
            # cv2.imshow('first_frame', first_frame)
            # cv2.imshow('Camera', blurred)
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
    subprocess.Popen(wsl_command, shell=True)

    main(f, 1)
