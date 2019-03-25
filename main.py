import os
import cv2
import sys
import argparse
import numpy as np
import pre_process as pp
from face_util.helpers import view_image
from face_util.helpers import load_image


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not; in case of preprocessing only one class is used")
	ap.add_argument("-m", "--modelPredictor", required=False, help="which model to use as face predictor in pre-processing")
	args = vars(ap.parse_args())

	# -----------------------------------------------------------------------------
	# 1. load the pre processed image in RGB 
	lt_dir = []
	if args["preProc"] == "1":
		print("Preprocessing images...")
		for d in os.listdir("./input"):
			lt_dir.append(d)
	else:
		for d in os.listdir("./pp_input"):
			lt_dir.append("./pp_input/"+d)

	if args["preProc"] == "0":
		data = load_image(lt_dir)

	# -----------------------------------------------------------------------------
	# 2. Do the pre-processing
	if args["preProc"] == "1":
		if args["modelPredictor"] == "dlib":
			for dir in lt_dir:
				print("\n---------------------Pre-processing ./input/"+dir+"...")
				data = load_image(["./input/"+dir])
				pre_process = pp.PreProcess(data)
				align_data = pre_process.align_resize_dlib()

				os.mkdir("./pp_input_dlib/"+dir)
				print("    Storing pre-processed image...")
				for i in range(len(align_data)):
					cv2.imwrite("./pp_input_dlib/"+dir+"/"+str(i+1)+".png", align_data[i])
		

		print("Pre-processing done")
		sys.exit()

# References:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/