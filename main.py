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


def get_accuracy(face_recognizer, x_test, y_test, rank):
	y_pred_cnt = 0

	for i in range(len(x_test)):
		img = x_test[i]
		res = cv2.face.StandardCollector_create()
		face_recognizer.predict_collect(img, res)
		res = res.getResults(sorted=True)
		# tmp = min(range,len(res))
		for j in range(rank):
			(x,y) = res[j]
			if x==y_test[i]:
				y_pred_cnt += 1
				break
	return (y_pred_cnt/len(y_test))


def cv2Eigen(x_train, x_test, y_train, y_test, rank):
	face_recognizer = cv2.face.EigenFaceRecognizer_create()
	face_recognizer.setNumComponents(80)
	face_recognizer.train(x_train, np.array(y_train))
	
	print('Accuracy cv2Eigen = ',get_accuracy(face_recognizer,x_test,y_test,rank))
	return

def cv2Fisher(x_train, x_test, y_train, y_test, rank):
	face_recognizer = cv2.face.FisherFaceRecognizer_create()
	face_recognizer.train(x_train, np.array(y_train))
	
	print('Accuracy cv2Fisher = ',get_accuracy(face_recognizer,x_test,y_test,rank))
	return

def cv2LBPH(x_train, x_test, y_train, y_test, rank):
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(x_train, np.array(y_train))
	
	print('Accuracy cv2LBPH = ',get_accuracy(face_recognizer,x_test,y_test,rank))
	return

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not; in case of preprocessing only one class is used")
	ap.add_argument("-m", "--modelPredictor", required=False, help="which model to use as face predictor in pre-processing")
	ap.add_argument("-n", "--inputPath", required=True, help="path to the folder to load image")
	ap.add_argument("-r", "--rank", required=True, help="rank for calculating accuracy")
	args = vars(ap.parse_args())

	# -----------------------------------------------------------------------------
	# 1. load the image in case of preprocessing
	lt_dir = []
	hm = args["inputPath"] # if the name(folder) is data then give => data/
	rank = int(args["rank"]) # rank while caculating accuracy
	if args["preProc"] == "1":
		print("Preprocessing images...")
		for d in os.listdir(hm):
			lt_dir.append(d)
		# lt_dir = ["Suraj"]
		# lt_dir = lt_dir[:2]

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
	# esc_lt = ["Ankush", "Anshuk", "Juhi", "Harsha_5th_year", "Naman", "Pragya", "Rachit", "Rakshith", "SaiPradeep", "Suraj"]
	esc_lt = ["Ankush", "Suraj"]
	for d in os.listdir(hm):
		if d not in esc_lt:
			lt_dir.append(hm+d)

	x_train = []
	x_test = []
	y_train = []
	y_test = []
	for i in range(len(lt_dir)):
		(data,y) = load_image([lt_dir[i]])
		pre_process = pp.PreProcess(data)
		data_gr = pre_process.get_grayscale(data)
		t_train,t_test,g_train,g_test = train_test_split(data_gr, y, test_size=0.2, random_state=30)
		x_train.extend(t_train)
		x_test.extend(t_test)
		y_train.extend([i]*len(g_train))
		y_test.extend([i]*len(g_test))

	del pre_process,data_gr,data,y,t_train,t_test,g_train,g_test

	# -----------------------------------------------------------------------------
	# 4. Making models and predicting
	cv2Eigen(x_train, x_test, y_train, y_test, rank)
	cv2Fisher(x_train, x_test, y_train, y_test, rank)
	cv2LBPH(x_train, x_test, y_train, y_test, rank)

# References:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# LBPH = 30, 31, 29 (random state values)
# ----------------------------------Accuracy----------------------------
# On pp_input_dlib
# Rank 1
# ------cv2Eigen =   0.7814569536423841
# ------cv2Fisher =  0.7086092715231788
# ------cv2LBPH =    0.9072847682119205
# Rank 3
# ------cv2Eigen =   0.8543046357615894
# ------cv2Fisher =  0.7814569536423841
# ------cv2LBPH =    0.9602649006622517
# Rank 10
# ------cv2Eigen =   0.9403973509933775
# ------cv2Fisher =  0.8410596026490066
# ------cv2LBPH =    0.9801324503311258
#
# On pp_input_harr_eye
# Rank 1
# ------cv2Eigen =   0.5384615384615384
# ------cv2Fisher =  0.4076923076923077
# ------cv2LBPH =    0.6538461538461539
# Rank 3
# ------cv2Eigen =   0.6538461538461539
# ------cv2Fisher =  0.47692307692307695
# ------cv2LBPH =    0.7692307692307693
# Rank 10
# ------cv2Eigen =   0.7846153846153846
# ------cv2Fisher =  0.6230769230769231
# ------cv2LBPH =    0.8384615384615385
#
