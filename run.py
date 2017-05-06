# -*- coding: utf-8 -*-
import cv2
import time
import sys
import numpy as np
import os


DEVICE_ID = 0
DEFAULT_SIZE = (320, 240)
FLIP = True

class Rect:
	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def modify(self):
		if self.w < 0:
			self.w *= -1
			self.x -= self.w
		if self.h < 0:
			self.h *= -1
			self.y -= self.h

class Meta:
	def __init__(self, window_name, img, rect):
		self.img = img
		self.img_bk =np.copy(img)
		self.rect = rect
		self.window_name = window_name

def mouse_event(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:		
		param.img = np.copy(param.img_bk)
		param.rect.x = x
		param.rect.y = y
	if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
		param.img = np.copy(param.img_bk)
		param.rect.w = x - param.rect.x
		param.rect.h = y - param.rect.y
		cv2.rectangle(param.img, (param.rect.x, param.rect.y), (param.rect.x + param.rect.w, param.rect.y + param.rect.h), (255, 255, 255), 2)
		cv2.imshow(param.window_name, param.img)
	if event == cv2.EVENT_LBUTTONUP:
		param.img = np.copy(param.img_bk)
		param.rect.w = x - param.rect.x
		param.rect.h = y - param.rect.y
		cv2.rectangle(param.img, (param.rect.x, param.rect.y), (param.rect.x + param.rect.w, param.rect.y + param.rect.h), (255, 255, 255), 2)
		cv2.imshow(param.window_name, param.img)

def get_frame(cap, size=DEFAULT_SIZE, flip=FLIP):
	res, frame = cap.read()
	if size is not None and len(size) == 2:
		frame = cv2.resize(frame, size)
	if flip is True:
		frame = frame[:,::-1]
	return frame

def get_gray_frame(cap, size=DEFAULT_SIZE, flip=FLIP):
	res, frame = cap.read()
	if size is not None and len(size) == 2:
		frame = cv2.resize(frame, size)
	if flip is True:
		frame = frame[:,::-1]
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	return gray

def start_preview(device_id = DEVICE_ID):
	#init camera device
	cap = cv2.VideoCapture(device_id)
	while True:
		start = time.time()
		# get frame
		#frame = get_frame(cap)
		frame = get_gray_frame(cap)

		# display frame
		cv2.imshow('camera preview', frame)
		if cv2.waitKey(1) == 27: # wait 1msec / finish by ESC key
			break

		elapsed_time = time.time() - start
		sys.stdout.write('elapsed_time {:3.3f} [s] \r'.format(1 / elapsed_time))
		sys.stdout.flush()

	# destroy window
	cv2.destroyAllWindows()
	#release camera device
	cap.release()

def configure_detect_rectangle(cap):
	crt_frame = get_gray_frame(cap)
	window_name = 'configure detect rectangle'
	detect_rect = Rect(0, 0, 0, 0)
	meta = Meta(window_name, crt_frame, detect_rect)
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.setMouseCallback(window_name, mouse_event, meta)
	cv2.imshow(window_name, crt_frame)
	while True:
		if cv2.waitKey(1) == 27: # wait 1msec / finish by ESC key
			break
	cv2.destroyAllWindows()
	return detect_rect, meta.img

def check_detect(b_frame, detect_rect):
	detect_rect.modify()
	window = b_frame[detect_rect.y : detect_rect.y + detect_rect.h, detect_rect.x : detect_rect.x + detect_rect.w]
	#check change ratio of binary values
	ratio = np.mean(window) / 255
	if ratio > 0.2:
		return True, ratio
	return False, ratio

def detector(cap, detect_rect):
	detect = False
	prev_frame = None
	crt_frame  = None
	while True:
		start = time.time()
		# get frame
		#frame = get_frame(cap)
		crt_frame = get_gray_frame(cap)

		if prev_frame is not None:
			# diff frame
			diff_frame = cv2.absdiff(crt_frame, prev_frame)
			# binary frame
			diff_b_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1]
			cv2.imshow('processing preview', diff_b_frame)
			detect, ratio = check_detect(diff_b_frame, detect_rect)
		if detect:
			# destroy window
			cv2.destroyAllWindows()
			return crt_frame

		# display frame
		cv2.imshow('camera preview', crt_frame)
		if cv2.waitKey(250) == 27: # wait 250 msec / finish by ESC key
			break

		prev_frame = crt_frame
		elapsed_time = time.time() - start
		sys.stdout.write('elapsed_time {:3.3f} [s] \r'.format(1 / elapsed_time))
		sys.stdout.flush()

	# destroy window
	cv2.destroyAllWindows()
	return None


def save_image(path, fname, img):
	if os.path.exists(path) == False:
		os.makedirs(path)
	cv2.imwrite(os.path.join(path, fname), img)

def start_motion_detector(device_id = DEVICE_ID):
	# init camera device
	cap = cv2.VideoCapture(device_id)
	# set detect area
	detect_rect, initial_img = configure_detect_rectangle(cap)
	# pole for detect
	detect_img = detector(cap, detect_rect)
	# release camera device
	cap.release()
	# save image
	if initial_img is not None:
		save_image('./save', '0_initial_image_with_rect.png', initial_img)
	if detect_img is not None:
		save_image('./save', '1_detect_image.png', detect_img)
		cv2.rectangle(detect_img, (detect_rect.x, detect_rect.y), (detect_rect.x + detect_rect.w, detect_rect.y + detect_rect.h), (255, 255, 255), 2)
		save_image('./save', '2_detect_image_with_rect.png', detect_img)
		# notify to user
		# by mail / slack (for exmaple)

if __name__ == '__main__':
	print('start motion detector')
	start_motion_detector()

