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

import keras.callbacks
import threading



def do_testimage(epoch):
	print("do_testimage: started after epoch", epoch)
	
	# setup dataset configurations
	class InferenceConfig(nucleus.NuclConfig):
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
	
	inference_config = InferenceConfig()
	
	
	
	# prepare validation dataset (required for class names)
	dataset_val = nucleus.NuclDataset()
	dataset_val.load_data('./Nucleus/train/', 'eval')
	dataset_val.prepare()
	
	# make inference model
	model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='.')
	
	# load last weights
	model.load_weights(model.find_last(), by_name=True)
	
	# make list of test images
	testdir = "./testimages/"
	filenames = os.listdir(testdir)
	for filename in filenames:
		filepath = os.path.join(testdir, filename)
		if os.path.isfile(filepath):
			print("filepath =", filepath, "isfile")
			
			# load testimage
			print("do_testimage: loading testimage...")
			testimg = cv2.imread(filepath)
			
			# do detection in testimage
			print("do_testimage: detection in testimage...")
			r = model.detect([testimg], verbose=1)[0]
			print("do_testimage: shape of masks:", r['masks'].shape)
			visualize.display_instances(testimg, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, (testimg.shape[:2]), r['scores'], basename=filename, epoch=epoch)
			
	
	print("do_testimage: finished")


# setup dataset configuration
class TrainingConfig(nucleus.NuclConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

training_config = TrainingConfig()

# check
if training_config.STEPS_PER_EPOCH == 1:
	print("*" * 80)
	print()
	print("STEPS_PER_EPOCH == 1")
	print()
	print("*" * 80)
	



class CustomCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("#" * 80)
		do_testimage(epoch)
		print("#" * 80)


# prepare training dataset
dataset_train = nucleus.NuclDataset()
dataset_train.load_data('./Nucleus/train/', 'train')
dataset_train.prepare()

# prepare validation dataset
dataset_val = nucleus.NuclDataset()
dataset_val.load_data('./Nucleus/train/', 'eval')
dataset_val.prepare()

# make training model
model = modellib.MaskRCNN(mode="training", config=training_config, model_dir='.')

# load coco weight
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

#modellib.dodump(model)

# prep logfile
logfile = 'log.csv'
if os.path.exists(logfile):
	os.unlink(logfile)

csv_logger = keras.callbacks.CSVLogger(logfile, append=True)

ph1_epochs = 64
ph2_epochs = 0
#ph1_epochs = 1
#ph2_epochs = 3
epfile = open("epoch.txt", "w")
epfile.write("ph1_epochs = " + str(ph1_epochs) + "\n")
epfile.write("ph2_epochs = " + str(ph2_epochs) + "\n")
epfile.close()

# trainig phase 1
print("init train phase 1")
model.train(dataset_train, dataset_val, learning_rate=training_config.LEARNING_RATE, epochs=ph1_epochs, layers='heads', custom_callbacks=[CustomCallback(), csv_logger])
print("exit train phase 1")


# trainig phase 2
print("init train phase 2")
model.train(dataset_train, dataset_val, learning_rate=training_config.LEARNING_RATE / 10, epochs=ph2_epochs, layers='all', custom_callbacks=[CustomCallback(), csv_logger])
print("exit train phase 2")

print("finished all training")







#######################################################################################################################################
## generate report
#
## make list of testimages
#lst0 = [f for f in os.listdir('testimages') if os.path.isfile(os.path.join('testimages', f))]  # testimage without path
#lst1 = []
#for i in range(len(lst0)):
#	lst1.append(os.path.join("testimages", lst0[i]))
#print("lst1 =", lst1)  # testimages with path
#
## make list of last instance image per testimage
#lst2 = []
#for i in range(len(lst0)):
#	base = os.path.splitext(lst0[i])[0]  # name without ext
#	ltmp = [f for f in os.listdir(os.path.join('testimages/instances', base)) if os.path.isfile(os.path.join('testimages/instances', base, f))]
#	ltmp.sort()
#	lastinst = ltmp[len(ltmp) - 1]
#	lst2.append(os.path.join('testimages/instances', base, lastinst))
#print("lst2 =", lst2)
#
#
## read images in lst1 and lst2 and check dimensions
#img1 = [Image.open(i) for i in lst1]
#for i in range(len(img1)):
#	print(img1[i].size)
#
#img2 = [Image.open(i) for i in lst2]
#for i in range(len(img2)):
#	print(img2[i].size)
#
## paste downscaled images into one imge
#num = len(img1)
#dim = 512
#neww = dim * 2 + 3
#newh = dim * num + num + 1
#newim = Image.new('RGB', (neww, newh), (255, 255, 255))
#
#
#offset = 0
#for i in range(len(img1)):
#	
#	im1 = img1[i].resize((dim, dim))
#	im2 = img2[i].resize((dim, dim))
#	
#	newim.paste(im1, (1, offset + 1))
#	newim.paste(im2, (dim + 2, offset + 1))
#	offset += dim + 1
#
#print("report saved as: collage.png")
#newim.save("collage.png")
#print("done")
threading.enumerate()

exit(0)















