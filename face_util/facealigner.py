from .helpers import FACIAL_LANDMARKS_68_IDXS
from .helpers import FACIAL_LANDMARKS_5_IDXS
from .helpers import shape_to_np
import numpy as np
import cv2

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.25, 0.25), desiredFaceWidth=256, desiredFaceHeight=None):
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align_eye(self, image, leftEyeCenter, rightEyeCenter):
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
		return output

	def align(self, image, gray, rect):
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		
		if (len(shape)==68):
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		else:
			(lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
			
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		return self.align_eye(image, leftEyeCenter, rightEyeCenter)