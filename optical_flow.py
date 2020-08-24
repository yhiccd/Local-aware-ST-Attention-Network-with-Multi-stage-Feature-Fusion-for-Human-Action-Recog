# Optical Flow Extraction
# Code provided - URL: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

import cv2 as cv
import numpy as np
import random
import torch
import torchvision
import os

def extractFrames(cap):
	'''
	Extracts the frames of a given video.

	Arguments:
	cap -- VideoCapture of the video being processed
	'''

	frames = []
	
	# Desired dimensions
	height = 300
	width = 224
	dim = (width, height)

	while True:
		ret, frame = cap.read()
		if ret:
			# Resize frame to be 300 x 224
			frame = cv.resize(frame, dim, interpolation = cv.INTER_AREA)

			# Add frame to the list of frames
			frames.append(frame)

			k = cv.waitKey(30) & 0xff
			if k == 27:
				break
		else:
			break

	cv.destroyAllWindows()
	cap.release()
	return frames

def selectInputs(frames, L = 10):
	'''
	Selects the desired inputs for the architecture: 1 RGB frame, 30 optical flow frames

	Arguments:
	frames 	-- list of RGB frames for a video
	L 		-- number of frames per set (default: 10 frames/set)

	Returns:
	t 			-- selected RGB frame
	set1_start 	-- start of the first set of optical flow frames
	set1_end 	-- end of the first set of optical flow frames
	set2_start 	-- start of second set
	set2_end 	-- end of second set
	set3_start 	-- start of third set
	set3_end 	-- end of third set
	'''
	numFrames = len(frames)

	# tau = random number from 1 to 10
	tau = random.randint(1, 5)

	# Choose value of t
	t = numFrames // 2		# t chosen to be in the middle

	# Chosen optical flow frames
	# First set
	set1_start = t - tau
	set1_end = set1_start + L
	# Second set
	set2_start = t
	set2_end = set2_start + L
	# Third set
	set3_start = t + tau
	set3_end = set3_start + L

	return t, set1_start, set1_end, set2_start, set2_end, set3_start, set3_end

def computeFlow(frames, start, end, f_params, lk, filename, num):
	'''
	Computes the optical flow of a given range of frames

	Arguments:
	frames		-- List of RGB frames
	start		-- Start of set
	end			-- End of set
	f_params	-- ShiTommasi feature parameters
    lk          -- Lucas Kanade optical flow parameters
    filename    -- Save filename
    num         -- Starting frame number

	Returns:
	flow	-- Tensor of optical flow
	'''

	color = (0, 255, 0)		# Green color

	# Take the first frame and find corners in it
	old_frame = frames[start]
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	# Used to decide points to track
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **f_params)

	flow = []

	for x in range(start, end):
		frame = frames[x]
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Calculate opticl flow
		# Pass in previous frame, previous points, and next frame
		# Returns next points + status numbers: 1 - found, 0 - not found
		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk)

		# Select good points
		good_new = p1[st==1]
		good_old = p0[st==1]

		# Create a mask image
		mask = np.zeros_like(old_frame)

		# Draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv.arrowedLine(mask, (a, b), (c, d), color, 2)
			frame = cv.circle(frame, (a, b), 5, color, -1)

		img = cv.add(frame, mask)
		fimg = cv.flip(img, 1)
        # Save the frames
		cv.imwrite(os.path.join(filename, "flow{}.jpg".format(str(num))), img)
		cv.imwrite(os.path.join(filename, "flip{}.jpg".format(str(num))), fimg)
		num += 1

		tensor = torch.from_numpy(img)
		flow.append(tensor)

		# Update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)

	return flow

def extractOptFlow(frames, filename):
	'''
	Extract the optical flow of selected frames

	Arguments:
	frames	    -- list of RGB frames
    filename    -- Save filename

	Returns:
	rgb		-- selected RGB frame
	flow1	-- first set of optical flow frames
	flow2	-- second set of optical flow frames
	flow3	-- third set of optical flow frames
	'''
	# Get selected frames
	rgb, s1_s, s1_e, s2_s, s2_e, s3_s, s3_e = selectInputs(frames)
	rgb = frames[rgb]
	# frgb = horizontal flipped
	frgb = cv.flip(rgb, 1)

	# Save the rgb frame
	cv.imwrite(os.path.join(filename, "rgb1.jpg"), rgb)
	cv.imwrite(os.path.join(filename, "rgb2.jpg"), frgb)

	# Parameters for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )

	# Parameters for Lucas Kanade optical flow
	lk_params = dict( winSize = (15, 15),
					  maxLevel = 2,
					  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


	# Compute the optical flows
	flow1 = computeFlow(frames, s1_s, s1_e, feature_params, lk_params, filename, 1)
	flow2 = computeFlow(frames, s2_s, s2_e, feature_params, lk_params, filename, 11)
	flow3 = computeFlow(frames, s3_s, s3_e, feature_params, lk_params, filename, 21)

	return rgb, flow1, flow2, flow3

def getInputs(cap, filename):
	'''
	Gets the input for the model

	Arguments:
	cap    	    -- VideoCapture of video
    filename    -- Save filename

	Returns:
	rgb		-- RGB frame
	flow1	-- first set of optical flow frames
	flow2	-- second set of optical flow frames
	flow3	-- third set of optical flow frames
	'''

	frames = extractFrames(cap)
	rgb, flow1, flow2, flow3 = extractOptFlow(frames, filename)

	return rgb, flow1, flow2, flow3
