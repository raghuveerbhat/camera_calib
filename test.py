import numpy as np
import cv2

from utils import *

videoP = '/home/raghu/comma_ai/calib_challenge/labeled/0.hevc'
cap = cv2.VideoCapture(videoP)


feature_params = dict(maxCorners=0,  # no limit on number of corners
                      qualityLevel=0.05,
                      minDistance=10,
                      blockSize=7)

lk_params = dict(winSize = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

ret, prev_frame = cap.read()

itval = 0
lines = []
while(cap.isOpened()):
	ret, frame = cap.read()

	if itval % 50 == 0:
		mask = np.zeros_like(prev_frame)
	itval += 1
	corners = shiTomasi(prev_frame, feature_params)
	if corners is not None and len(corners) != 0:
		good_old, good_new = opticalFlow(prev_frame, frame, corners, lk_params)
		prev_frame = frame.copy()
		corners = np.reshape(corners, (-1, 2)).astype(int)
		corner_mask = frame.copy()
		for i, (new, old) in enumerate(zip(good_new,good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			lines.append((a, b, c, d))
			mask = cv2.line(mask, (a, b), (c, d), (255, 0, 255), thickness=1)
			corner_mask = cv2.circle(corner_mask, center=(a, b), radius=2, color=(255, 255, 255), thickness=-1)
		for corner in corners:
			corner_mask = cv2.circle(corner_mask, center=(corner[0], corner[1]), radius=2, color=(0, 255, 0), thickness=-1)
		# frame = cv2.add(frame, corner_mask)
		frame = cv2.add(corner_mask, mask)
		corners = good_new.reshape(-1, 1, 2).astype(np.float32)

		fld = cv2.ximgproc.createFastLineDetector()
	    # Get line vectors from the image
		lines_fd = fld.detect(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
	    # Draw lines on the image
		frame = fld.drawSegments(frame, lines_fd)

		# intersection_point, intersection_lines = get_biggest_intersection(np.asarray(lines), intersection_threshold=3)
		# frame = cv2.circle(frame, center=(int(intersection_point[0]),int(intersection_point[1])), radius=5, color=(255, 255, 0), thickness=-1)
		
		lines = []
	else:
		prev_frame = frame.copy()
	cv2.imshow("Test", frame)
	if cv2.waitKey(delay=1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
