import time
import os
import sys
import math
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import nucleus

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

TRAINED_MODEL_PATH = "mask_rcnn_nucl_dataset_0030.h5"
TEST_IMG_DIR = "./testimages/"

def do_testimage():
	print("do_testimage: started", file=sys.stderr)
	
	# setup dataset configurations
	class InferenceConfig(nucleus.NuclConfig):
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
	
	inference_config = InferenceConfig()
	
	
	# setup class_names
	class_names = ["BG", "nucl"]
	
	# make inference model
	model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='.')
	
	# load last weights
	model.load_weights(TRAINED_MODEL_PATH, by_name=True)
	
	# check test images are named correctly
	print("CHECK TEST IMAGE FILENAMES", file=sys.stderr)
	filenames = os.listdir(TEST_IMG_DIR)
	for filename in filenames:
		print("filename =", filename, file=sys.stderr)
		name, ext = os.path.splitext(filename)
		if not name.isnumeric():
			print("name is not numeric:", filename, file=sys.stderr)
			exit()
	
	print("#" * 80, file=sys.stderr)
	
	# make list of test images
	filenames = os.listdir(TEST_IMG_DIR)
	for filename in filenames:
		filepath = os.path.join(TEST_IMG_DIR, filename)
		if os.path.isfile(filepath):
			# load testimage
			testimg = cv2.imread(filepath)
			
			# do detection in testimage
			print("do_testimage: detection in testimage...", file=sys.stderr)
			r = model.detect([testimg], verbose=1)[0]
			
			# swap width and height of testimg
			testimg_newshape = (testimg.shape[:2][1], testimg.shape[:2][0])
			
			
			# detect and remove instances at boundaries
			mindist = 20  # minimal distance to boundary
			img_width = testimg.shape [1]
			img_height = testimg.shape [0]
			nelem = len(r['rois'])
			for i in range(nelem):
				idx = nelem - i - 1
				print (idx, r['rois'][idx])
				if r['rois'][idx][0] < mindist or r['rois'][idx][1] < mindist or r['rois'][idx][2] >= (img_height - mindist) or r['rois'][idx][3] >= (img_width - mindist):
					print ("removing", idx, ":", r['rois'][idx])
					r['rois'] = np.delete (r['rois'], idx, 0)
					r['class_ids'] = np.delete (r['class_ids'], idx, 0)
					r['scores'] = np.delete (r['scores'], idx, 0)
					r['masks'] = np.delete (r['masks'], idx, 2)
			# <end of detect and remove instances at boundaries>
			
			
			visualize.display_instances_smai2(testimg, r['rois'], r['masks'], r['class_ids'], class_names, testimg_newshape, r['scores'], basename=filename)
			print("-" * 80, file=sys.stderr)
	
	print("do_testimage: finished", file=sys.stderr)


do_testimage()

