import numpy as np
import cv2

import matplotlib.pyplot as plt

from utils import *

videoP = '/home/raghu/comma_ai/calib_challenge/labeled/0.hevc'
cap = cv2.VideoCapture(videoP)

focal_length = 910.0

frame_size = (1164, 874)

cam_intrinsics = np.array([
  [focal_length, 0.0, float(frame_size[0])/2],
  [0.0, focal_length, float(frame_size[1])/2],
  [0.0, 0.0, 1.0]])

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
angles = []
while(cap.isOpened()):
	ret, frame = cap.read()

	

	fld = cv2.ximgproc.createFastLineDetector()
    # Get line vectors from the image
	lines_fd = fld.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	lines = lines_fd.reshape(lines_fd.shape[0],4)
	for i, line in enumerate(lines):
		angles.append(get_line_angle_degree(line))
    # Draw lines on the image
	frame = fld.drawSegments(frame, lines_fd)
	
	np.random.seed(19680801)

	# example data
	mu = 100  # mean of distribution
	sigma = 15  # standard deviation of distribution
	x = angles
	num_bins = 18

	fig, ax = plt.subplots()

	# the histogram of the data
	n, bins, patches = ax.hist(x, num_bins, density=True)

	# add a 'best fit' line
	y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
	     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
	ax.plot(bins, y, '--')
	ax.set_xlabel('Smarts')
	ax.set_ylabel('Probability density')
	ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()
	plt.show()


	angles = []
	# intersection_point, intersection_lines = get_biggest_intersection(np.asarray(lines), intersection_threshold=3)
	# frame = cv2.circle(frame, center=(int(intersection_point[0]),int(intersection_point[1])), radius=5, color=(255, 255, 0), thickness=-1)
	cv2.imshow("Test", frame)
	if cv2.waitKey(delay=10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
