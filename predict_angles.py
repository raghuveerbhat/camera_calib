import math
import numpy as np
import cv2
from utils import *


videoP = '/Users/raghuveerbhat/commai/calib_challenge/labeled/3.hevc'
cap = cv2.VideoCapture(videoP)

focal_length = 910.0

frame_size = (1164, 874)

K = np.array([
  [focal_length, 0.0, float(frame_size[0])/2],
  [0.0, focal_length, float(frame_size[1])/2],
  [0.0, 0.0, 1.0]])

inv_K = np.linalg.inv(K)

#hough params
canny_tsigma = 0.25
min_points=0.075
min_line_length=0.4
max_line_gap=0.2


DRAW_PALETTE = [
   (0, 171, 169), (255, 0, 151), (162, 0, 255), (27, 161, 226), (240, 150, 9),
   (0, 102, 101), (153, 0, 90), (97, 0, 153), (16, 96, 135), (144, 90, 5),
   (102, 204, 203), (255, 102, 192), (199, 102, 255), (118, 198, 237), (246, 192, 107),
]
 
def draw_lines(lines, dest_image, color=(0, 0, 255), thickness=2):
   for x1, y1, x2, y2 in lines:
       cv2.line(dest_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
 
def draw_line_groups(line_groups, dest_image, color_options=DRAW_PALETTE):
   for i, lines in enumerate(line_groups):
       if i > len(color_options):
           print("Warning: There are fewer color options than groups.")
           color = (0, 0, 0)
       else:
           color = DRAW_PALETTE[i]
       draw_lines(lines, dest_image, color=color)
 
def draw_multiple_lines(lines, dest_image):
   for i, lines in enumerate(lines):
       color = (255, 0, 151)
       draw_lines(lines, dest_image, color=color)

def draw_point(point, dest_image):
  cv2.circle(dest_image, center=(int(point[0]), int(point[1])), radius=10, color=(0, 255, 255), thickness=-1)

point_tracker = []

def houghDetection():
   while(cap.isOpened()):
       ret, frame = cap.read()
       if ret == False:
           break
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       #Enhance edges in the image
       grayb = cv2.GaussianBlur(gray, (5, 5), 0)
       #Detect edges in the image
       v = np.median(gray)
       lower = int(max(0, (1.0 - canny_tsigma) * v))
       upper = int(min(255, (1.0 + canny_tsigma) * v))
       edged_image = cv2.Canny(grayb, lower, upper)
 
       height, width = edged_image.shape[:2]
       size = math.sqrt(height ** 2 + width ** 2)
       lines = cv2.HoughLinesP(
           edged_image,
           1,
           np.pi / 180,
           math.ceil(min_points * size),
           minLineLength=math.ceil(min_line_length * size),
           maxLineGap=math.ceil(max_line_gap * size))
       if lines is None:
           continue
       #print(lines.shape)
       lines_hl = lines.reshape(lines.shape[0], 4)
       print(lines_hl.shape)
       intersection_point, intersection_lines = get_biggest_intersection(lines_hl[:], intersection_threshold=3)
       if intersection_point is not None:
           point_tracker.append(intersection_point)
       else:
           intersection_point = point_tracker[-1]
       draw_point(intersection_point, frame)
       #draw_multiple_lines(lines, frame)
       cv2.imshow("sparse optical flow", frame)
       # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
       if cv2.waitKey(10) & 0xFF == ord('q'):
           break

houghDetection()

def print_angles(point_tracker):
    for i,point in enumerate(point_tracker):
        vpz = np.array([point[0], point[1], 1])
        r3 = np.dot(inv_K, vpz)
        r3 = r3/np.linalg.norm(r3)
        alpha = np.arctan(r3[0]/r3[2])
        scientific_notation = "{:e}".format(alpha)
        beta = np.arccos(r3[1])
        print(scientific_notation)

print_angles(point_tracker)