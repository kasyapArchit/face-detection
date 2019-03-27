import numpy as np
import cv2
import os
from collections import OrderedDict

def load_image(lt_dir):
	print("Loading image...")
	res = []
	y = []
	
	for i in range(len(lt_dir)):
		d = lt_dir[i]
		print("    "+str(i+1)+" "+d+"...")
		
		for pth in os.listdir(d):
			img = cv2.imread(os.path.join(d, pth), cv2.IMREAD_COLOR)
			res.append(img)
			y.append(i)

	print("Loading complete")
	return (res, y)

def view_image(img_lt):
	print("Viewing images...")
	
	for i in range(len(img_lt)):
		img = cv2.resize(img_lt[i], (256,256))
		cv2.imshow(str(i), img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Viewed all images")

# -------------------------------------------------------------------------------
#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords








