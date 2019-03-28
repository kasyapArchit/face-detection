import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
import pre_process as pp
from face_util.helpers import view_image
from face_util.helpers import load_image
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def cv2Eigen(x_train, x_test, y_train, y_test):
	face_recognizer = cv2.face.EigenFaceRecognizer_create()
	face_recognizer.train(x_train, np.array(y_train))
	y_pred = []
	for img in x_test:
		label, confidence = face_recognizer.predict(img)
		y_pred.append(label)
	
	print('Accuracy cv2Eigen = ' + str(accuracy_score(y_test, y_pred)))
	return

def cv2Fisher(x_train, x_test, y_train, y_test):
	face_recognizer = cv2.face.FisherFaceRecognizer_create()
	face_recognizer.train(x_train, np.array(y_train))
	y_pred = []
	for img in x_test:
		label, confidence = face_recognizer.predict(img)
		y_pred.append(label)
	
	print('Accuracy cv2Fisher = ' + str(accuracy_score(y_test, y_pred)))
	return

def cv2LBPH(x_train, x_test, y_train, y_test):
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(x_train, np.array(y_train))
	y_pred = []
	for img in x_test:
		label, confidence = face_recognizer.predict(img)
		y_pred.append(label)
	
	print('Accuracy cv2LBPH = ' + str(accuracy_score(y_test, y_pred)))
	return

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not; in case of preprocessing only one class is used")
	ap.add_argument("-m", "--modelPredictor", required=False, help="which model to use as face predictor in pre-processing")
	ap.add_argument("-n", "--inputPath", required=True, help="path to the folder to load image")
	args = vars(ap.parse_args())

	# -----------------------------------------------------------------------------
	# 1. load the image in case of preprocessing
	lt_dir = []
	hm = args["inputPath"] # if the name(folder) is data then give => data/
	if args["preProc"] == "1":
		print("Preprocessing images...")
		for d in os.listdir(hm):
			lt_dir.append(d)
		# lt_dir = ["Suraj"]
		# lt_dir = lt_dir[:2]
	else:
		# esc_lt = ["Ankush", "Anshuk", "Juhi", "Harsha_5th_year", "Naman", "Pragya", "Rachit", "Rakshith", "SaiPradeep", "Suraj"]
		esc_lt = ["Ankush", "Suraj"]
		for d in os.listdir(hm):
			if d not in esc_lt:
				lt_dir.append(hm+d)

	if args["preProc"] == "0":
		(data,y) = load_image(lt_dir)

	# -----------------------------------------------------------------------------
	# 2. Do the pre-processing
	if args["preProc"] == "1":
		if args["modelPredictor"] == "dlib":
			for dir in lt_dir:
				print("\n---------------------Pre-processing "+hm+"/"+dir+"...")
				(data,y) = load_image([hm+dir])
				pre_process = pp.PreProcess(data)
				align_data = pre_process.align_resize_dlib()
				# view_image(align_data)
				os.mkdir("./pp_input_dlib/"+dir)
				print("    Storing pre-processed image...")
				for i in range(len(align_data)):
					cv2.imwrite("./pp_input_dlib/"+dir+"/"+str(i+1)+".png", align_data[i])
		else:
			for dir in lt_dir:
				print("\n---------------------Pre-processing "+hm+"/"+dir+"...")
				(data,y) = load_image([hm+dir])
				pre_process = pp.PreProcess(data)
				align_data = pre_process.align_resize_harr()
				# view_image(align_data)
				os.mkdir("./pp_input_harr/"+dir)
				print("    Storing pre-processed image...")
				for i in range(len(align_data)):
					cv2.imwrite("./pp_input_harr/"+dir+"/"+str(i+1)+".png", align_data[i])

		print("Pre-processing done")
		sys.exit()


	# -----------------------------------------------------------------------------
	# 3. train test split and shuffling
	pre_process = pp.PreProcess(data)
	data_gr = pre_process.get_grayscale(data)
	del pre_process
	x_train,x_test,y_train,y_test = train_test_split(data_gr, y, test_size=0.2, random_state=29)

	# -----------------------------------------------------------------------------
	# 4. Making models and predicting
	cv2Eigen(x_train, x_test, y_train, y_test)
	cv2Fisher(x_train, x_test, y_train, y_test)
	cv2LBPH(x_train, x_test, y_train, y_test)

# References:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/