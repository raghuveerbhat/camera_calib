import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

videoP = '/Users/raghuveerbhat/commai/calib_challenge/labeled/0.hevc'
cap = cv2.VideoCapture(videoP)


feature_params = dict(maxCorners=0,  # no limit on number of corners
                      qualityLevel=0.05,
                      minDistance=10,
                      blockSize=7)

lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

ret, frame1 = cap.read()
gframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_keypoints = fast.detect(gframe1, None)
frame1_keypoints, frame1_descriptor = brief.compute(gframe1, frame1_keypoints)
mask = np.zeros_like(frame1)
itval = 0
lines = []
while(cap.isOpened()):
	ret, frame2 = cap.read()
	gframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	itval += 1

	# gframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	# orb = cv2.ORB_create()

	# frame1_keypoints, frame1_descriptor = orb.detectAndCompute(frame1, None)
	# frame2_keypoints, frame2_descriptor = orb.detectAndCompute(frame2, None)

	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	# matches = bf.match(frame1_descriptor, frame2_descriptor)
	# matches = sorted(matches, key = lambda x : x.distance)

	# if itval % 3 == 0:
	# 	mask = np.zeros_like(frame1)
	
	if itval % 100 == 0:
		print("hgjghjg")
		frame1_keypoints = fast.detect(gframe1, None)
		frame1_keypoints, frame1_descriptor = brief.compute(gframe1, frame1_keypoints)
	frame2_keypoints = fast.detect(gframe2, None)
	frame2_keypoints, frame2_descriptor = brief.compute(gframe2, frame2_keypoints)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(frame1_descriptor, frame2_descriptor)
	matches = sorted(matches, key=lambda x: x.distance)

	src_pts = np.float32(
		[frame1_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32(
		[frame2_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()
	print(src_pts.shape)
	print(dst_pts.shape)
	#cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
	src_pts = src_pts[np.array(matchesMask) == 1]
	src_pts = src_pts.reshape(src_pts.shape[0], 2)
	dst_pts = dst_pts[np.array(matchesMask) == 1]
	dst_pts = dst_pts.reshape(dst_pts.shape[0], 2)

	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
 #                   singlePointColor = None,
 #                   matchesMask = matchesMask, # draw only inliers
 #                   flags = 2)
	# result = cv2.drawMatches(frame1, frame1_keypoints, frame2, frame2_keypoints, matches, None, **draw_params)
	for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
		a, b = src[0], src[1]
		c, d = dst[0], dst[1]
		dist = point_to_point_dist((a, b), (c, d))
		if dist != 0:
			lines.append((a, b, c, d))
		if dist<20:
			result = cv2.line(frame1, (int(a), int(b)),
                    (int(c), int(d)), (255, 0, 255), thickness=1)
	intersection_point, intersection_lines = get_biggest_intersection(
		lines[:50], intersection_threshold=3)
	result = cv2.circle(result, center=(int(intersection_point[0]), int(
		intersection_point[1])), radius=3, color=(255, 255, 255), thickness=-1)
	lines = []
	frame1 = frame2.copy()
	frame1_keypoints = frame2_keypoints
	frame1_descriptor = frame2_descriptor
	cv2.imshow("Test", result)
	if cv2.waitKey(delay=1) & 0xFF == ord('q'):
		break
