import pre_process as pp
import numpy as np
import cv2
import os
import argparse
from face_util.helpers import view_image
from face_util.helpers import load_image


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", required=True, help="whether to use main dataset")
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not")
	args = vars(ap.parse_args())

	# -----------------------------------------------------------------------------
	# 1. load the image in RGB
	lt_dir = []
	if args["data"] == "0":
		lt_dir = ["temp"]
	else:
		for d in os.listdir("./input"):
			lt_dir.append("./input/"+d)
	
	data = load_image(lt_dir, args)
	if args["data"] == "0":
		view_image(data)

	# -----------------------------------------------------------------------------
	# 2. Do the pre-processing
	if args["preProc"] == "1":
		pre_process = pp.PreProcess(data)
		align_data = pre_process.align_resize()
		if args["data"] == "0":
			view_image(align_data)

# References:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/