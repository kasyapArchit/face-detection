import numpy as np
import cv2
import dlib
from face_util.facealigner import FaceAligner
from face_util.helpers import view_image
from face_util.helpers import rect_to_bb

class PreProcess:
	def __init__(self, data):
		self.data = data
		self.n = len(data)

	def get_grayscale(self, dat):
		res = []
		for img in dat:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			res.append(img)
		return res

	def resize(self, dat, width=None, height=None, inter=cv2.INTER_AREA):
		res = []
		
		for image in dat:
			dim = None
			(h, w) = image.shape[:2]

			if width is None and height is None:
				res.append(img)
				continue

			if width is None:
				r = height / float(h)
				dim = (int(w * r), height)
			else:
				r = width / float(w)
				dim = (width, int(h * r))

			img = cv2.resize(image, dim, interpolation=inter)
			res.append(img)

		return res

	def align_resize(self):
		print("Aligning and resizing images...")
		res = []
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		fa = FaceAligner(predictor, desiredFaceWidth=256)
		
		resized = self.resize(self.data, width=800)
		gray = self.get_grayscale(resized)

		for i in range(self.n):
			print("    "+str(i+1))
			image = resized[i]
			rects = detector(gray[i], 2)
			for rect in rects:
				(x, y, w, h) = rect_to_bb(rect)
				# faceOrig = self.resize([image[y:y + h, x:x + w]], width=256)[0]
				faceAligned = fa.align(image, gray[i], rect)
			
			res.append(faceAligned)
		print("Images aligned")
		return res
