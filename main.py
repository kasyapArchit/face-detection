import pre_process as pp
import numpy as np
import cv2
import os
import argparse


def load_image(lt_dir):
	print("Loading image...")
	res = []
	
	for i in range(len(lt_dir)):
		d = lt_dir[i]
		print("    "+str(i+1)+" "+d+"...")
		
		for pth in os.listdir(d):
			img = cv2.imread(os.path.join(d, pth), cv2.IMREAD_COLOR)
			# if the image size(memory) is too large then resize the image (memory trade of)
			if args["data"]!="0":
				img = cv2.resize(img, (200,200))
			res.append(img)

	print("Loading complete")
	return res

def view_image(img_lt):
	print("Viewing images...")
	
	for i in range(len(img_lt)):
		img = cv2.resize(img_lt[i], (256,256))
		cv2.imshow(str(i), img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Viewed all images")

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", required=True, help="whether to use main dataset")
	ap.add_argument("-p", "--preProc", required=True, help="to do pre-processing or not")
	args = vars(ap.parse_args())

	# 1. load the image in RGB
	lt_dir = []
	if args["data"] == "0":
		lt_dir = ["temp"]
	else:
		for d in os.listdir("./input"):
			lt_dir.append("./input/"+d)
	
	data = load_image(lt_dir)
	if args["data"] == "0":
		view_image(data)

	# -----------------------------------------------------------------------------
	# 2. Do the pre-processing
	if args["preProc"] == "1":
		pre_process = pp.PreProcess(data)