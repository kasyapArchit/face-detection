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

	def align_resize_dlib(self):
		print("Aligning and resizing images...")
		res = []
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		fa = FaceAligner(predictor, desiredFaceWidth=256)
		
		resized = self.resize(self.data, width=800)
		gray = self.get_grayscale(resized)

		for i in range(self.n):
			image = resized[i]
			rects = detector(gray[i], 2)
			if len(rects) == 0:
				print("    "+str(i+1)+" failed")
				continue
			print("    "+str(i+1))
			for rect in rects:
				# (x, y, w, h) = rect_to_bb(rect)
				# faceOrig = self.resize([image[y:y + h, x:x + w]], width=256)[0]
				faceAligned = fa.align(image, gray[i], rect)
			
			res.append(faceAligned)
		print("Images aligned")
		return res
	
	def align_resize_harr(self):
		print("Aligning and resizing images...")
		res = []
		face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
		fa = FaceAligner(eye_cascade, desiredFaceWidth=256)

		gray = self.get_grayscale(self.data)

		for i in range(self.n):
			img = self.data[i]
			img_gray = gray[i]
			
			try:
				faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
				(x,y,w,h) = faces[0]
				roi_gray = img_gray[y:y+h, x:x+w]
				roi_color = img[y:y+h, x:x+w]
				
				eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
				(ex1,ey1,ew1,eh1) = eyes[0]
				(ex2,ey2,ew2,eh2) = eyes[1]
				
				if(ex1 < ex2):
					lt_mid = [(ex1+(int)(ew1/2)), (ey1+(int)(eh1/2))]
					rt_mid = [(ex2+(int)(ew2/2)), (ey2+(int)(eh2/2))]
				else:
					rt_mid = [(ex1+(int)(ew1/2)), (ey1+(int)(eh1/2))]
					lt_mid = [(ex2+(int)(ew2/2)), (ey2+(int)(eh2/2))]
			except:
				print("    "+str(i+1)+" failed")
				continue
			print("    "+str(i+1))
			res.append(fa.align_eye(roi_color, rt_mid, lt_mid))
			# res.append(roi_color)
		return res
