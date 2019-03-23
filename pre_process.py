import numpy as np
import cv2

class PreProcess:
	data = []
	gray = []

	def __init__(self, data):
		self.data = data
		for img in self.data:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			self.gray.append(img)

	def get_grayscale(self):
		return self.gray

	def get_resized(self, h, w):
		res = []
		for img in self.data:
			res.append(cv2.resize(img, (h,w)))
		return res

	def align_resize(self):
		
		return
