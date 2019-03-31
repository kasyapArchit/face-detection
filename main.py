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
import my_pca 
from sklearn import svm
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV


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
	face_recognizer.setNumComponents(50)
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

def modify(data):
	a=[]
	for image in data:
		# print (image.shape)
		a.append(image.ravel())
	return np.array(a)

def my_pca_predict(x_train, x_test, y_train, y_test):
	X_train = modify(x_train)
	X_test = modify(x_test)
	# print (x_train.shape)

	pca = my_pca.PCA(n_components = 60)
	pca.fit(X_train)

	train = pca.transform(X_train)
	test = pca.transform(X_test)
	
	model = svm.SVC(kernel='linear', probability=True)
	model.fit(train, y_train)
	predict_prob = model.predict_proba(test)
	
	top_1 = 0
	top_3 = 0
	top_10 = 0
	total_len = len(y_test)
	for i in range(total_len):
		idx = np.argsort(-predict_prob[i])
		if y_test[i] in idx[:1]:
			top_1 += 1
		if y_test[i] in idx[:3]:
			top_3 += 1
		if y_test[i] in idx[:10]:
			top_10 += 1
	
	print("Top 1 accuracy", top_1/total_len)
	print("Top 3 accuracy", top_3/total_len)
	print("Top 10 accuracy", top_10/total_len)

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not; in case of preprocessing only one class is used")
	ap.add_argument("-m", "--modelPredictor", required=False, help="which model to use as face predictor in pre-processing")
	ap.add_argument("-g", "--gender", required=True, help="whether to do gender based prediction")
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
	if args["gender"] == "0":
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
	else:
		esc_lt = ["Ankush", "Suraj"]
		girls = [hm+"Anagha", hm+"Deepika", hm+"Deepti", hm+"Devyani", hm+"Juhi", hm+"Nehal", hm+"Prachi", hm+"Pragya", hm+"Shiloni", hm+"Sowmya", hm+"Sravya", hm+"Tripti"]
		for d in os.listdir(hm):
			if d not in esc_lt:
				lt_dir.append(hm+d)

		x = []
		y = []
		for i in range(len(lt_dir)):
			(data,_) = load_image([lt_dir[i]])
			pre_process = pp.PreProcess(data)
			data_gr = pre_process.get_grayscale(data)
			x.extend(data_gr)
			if lt_dir[i] in girls:
				y.extend([0]*len(data_gr))
			else:
				y.extend([1]*len(data_gr))
			
		x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=30)
		del data, data_gr, pre_process, x, y


	# -----------------------------------------------------------------------------
	# 4. Making models and predicting
	cv2Eigen(x_train, x_test, y_train, y_test, rank)
	cv2Fisher(x_train, x_test, y_train, y_test, rank)
	cv2LBPH(x_train, x_test, y_train, y_test, rank)
	my_pca_predict(x_train, x_test, y_train, y_test)


# References:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# LBPH = 30, 31, 29 (random state values)
