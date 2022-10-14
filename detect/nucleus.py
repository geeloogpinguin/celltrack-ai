import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from pathlib import Path
import skimage.draw
from numpy import asarray

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib


# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = './'

#
## Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#
#
## Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#	utils.download_trained_weights(COCO_MODEL_PATH)
#	

class NuclConfig(Config):
	
	NAME = "nucl_dataset"
	
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
	NUM_CLASSES = 1 + 1
	
	IMAGE_MIN_DIM = 128
	IMAGE_MAX_DIM = 512
	
	STEPS_PER_EPOCH = 50
	
	VALIDATION_STEPS = 5
	
	BACKBONE = 'resnet50'
	
	# To be honest, I haven't taken the time to figure out what these do
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
	TRAIN_ROIS_PER_IMAGE = 32
	MAX_GT_INSTANCES = 50 
	POST_NMS_ROIS_INFERENCE = 500 
	POST_NMS_ROIS_TRAINING = 1000 
	

class NuclDataset(utils.Dataset):
	# @source: https://www.kaggle.com/stargarden/m-r-cnn-matterport-1
	# @source: https://towardsdatascience.com/object-detection-using-mask-r-cnn-on-a-custom-dataset-4f79ab692f6d
	# @source: Mast_RCNN/samples/balloon/balloon.py
	
	def load_data(self, images_dir, subset):
		"""
			Load the coco-like dataset from json
			args:
				annotation_json: The path to the coco annotation json file
				images_dir: The directory holding the images referred by the json file
		"""
		# add the "nucl" category
		self.add_class("dataset", 1, "nucl")
		
		for sd in os.listdir(images_dir):
			if os.path.isdir(os.path.join(images_dir, sd)):
				
				if subset == "test":
					image_path = os.path.join(images_dir, sd, "images", sd) + ".png"
					#print(image_path)
					self.add_image("dataset", image_id=sd, path=image_path, images_dir=images_dir, subset=subset)
				else:
					if sd.startswith('7'):
						if subset == "eval":
							image_path = os.path.join(images_dir, sd, "images", sd) + ".png"
							#print(image_path)
							self.add_image("dataset", image_id=sd, path=image_path, images_dir=images_dir, subset=subset)
					else:
						if subset == "train":
							image_path = os.path.join(images_dir, sd, "images", sd) + ".png"
							#print(image_path)
							self.add_image("dataset", image_id=sd, path=image_path, images_dir=images_dir, subset=subset)
		
		
	
	def load_mask(self, image_id):
		"""
			Generate instance mask for an image
			returns:
				mask: a bool array of shape [height, width, instance count] with one mask per instance
				class_id: an ID array of class IDs of the instance masks
		"""
		info = self.image_info[image_id]
		iid = info['id']
		images_dir = info['images_dir']
		subset = info['subset']
		
		#print("$$$ load_mask $$$ for image_id =", image_id)
		#print("info =", info)
		#print("load_mask")
		#print("subset =", subset)
		
		masks_dir = os.path.join(images_dir, iid, "masks")
		#print("masks_dir =", masks_dir)
		
		if not os.path.isdir(masks_dir):
			# no masks for images
			#print("loading no mask")
			nomask = np.empty([0, 0, 0])
			noclassids = np.empty([0], np.int32)
			return nomask, noclassids
		
		
		
		masks = []
		for mask in os.listdir(masks_dir):
			mask_path = os.path.join(images_dir, iid, "masks", mask)
			#print("mask_path =", mask_path)
			image = Image.open(mask_path)
			dat = asarray(image)
			masks.append(dat)
			
		masks = np.stack(masks, axis=-1)
		
		return masks, np.ones([masks.shape[-1]], dtype=np.int32)



































