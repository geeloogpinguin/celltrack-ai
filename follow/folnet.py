import numpy as np
import sys
from timeit import default_timer as timer
#np.random.seed(1337)  # for reproducibility

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, Adadelta
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Parameters
input_dimx = 300  # breedte van breedste nucleus
input_dimy = 300  # hoogte van hoogste nucleus
lr = 0.000005


def get_datadir (cfgfile):
	f = open (cfgfile, "r")
	for line in f:
		words = line.split ("=")
		if (words[0].strip () == "datadir"):
			DD = words[1].strip ()
	return DD

#datadir = get_datadir ("../config.txt")
#print ("Found datadir is", datadir)



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def FAKE_loadnucl(filename): # for debugging only
	fin_w = 300
	fin_h = 300
	return [[0 for x in range (fin_w)] for y in range (fin_h)]
	
	
def loadnucl(filename):
	# load image and centre in upsized
	
	fin_w = 300
	fin_h = 300
	#print("fin_w =", fin_w, " fin_h =", fin_h)
	
	# Get x- and y-coordinate from filename
	xcrd = int (filename[-14:-10])
	ycrd = int (filename[-8:-4])
	#print ("X =", xcrd)
	#print ("Y =", ycrd)
	
	
	# Read image and get width and height
	print("folnet :: loadnucl. filename =", filename)
	img = Image.open(filename)
	w, h = img.size
	print("w =", w, " h =", h)
	
	# Get subregion if image is too large
	do_crop = False
	le = 0
	up = 0
	ri = w
	lo = h
	if (h > fin_h):  # Image too high, crop in height
		do_crop = True
		up = (h - fin_h) / 2
		lo = fin_h + up
	
	if (w > fin_w):  # Image too weight, crop in width
		do_crop = True
		le = (w - fin_w) / 2
		ri = fin_w + le
	
	if (do_crop):  # Crop image
		img = img.crop ((le, up, ri, lo))
		w, h = img.size
		#print ("cropping image named", filename, "to", w, h)
	
	
	
	# Calc offsets if image is too small (and requires enlargemant)
	xoffset = int((fin_w - w) / 2)
	yoffset = int((fin_h - h) / 2)
	
	
	print("xoffset =", xoffset, " yoffset =", yoffset)
	fin_mx = [[0 for y in range(fin_h)] for x in range(fin_w)]
	#imgplot = plt.imshow(img)
	px = img.load()
	
	for y in range(0, h):  # expensive
		for x in range(0, w):
			#G = px[x, y][1]  # use green channel only
			G = px[x, y]  # grayscale image
			fin_mx[xoffset + x][yoffset + y] = max(G)  # dunno why, but it works...
			#print("[", xoffset + x, "][", yoffset + y, "]")
	
	# put x coordinate in top banner
	for x in range (fin_w):
		fin_mx [x][1] = xcrd
		fin_mx [x][2] = xcrd
		fin_mx [x][3] = xcrd
	
	# put y coordinate in bottom banner
	for x in range (fin_w):
		fin_mx [x][fin_h - 2] = ycrd
		fin_mx [x][fin_h - 3] = ycrd
		fin_mx [x][fin_h - 4] = ycrd
	
	#dumpimgmx (fin_mx)
	
	fin_mx = np.asarray(fin_mx)
	
	#np.set_printoptions(threshold=sys.maxsize)
	#print("fin_mx =")
	#print(fin_mx)
	
	
	
	
	
	
	fin_mx = fin_mx.astype('float32')
	fin_mx /= 255
	
	return fin_mx


def dumpimgmx(imgmx):
	w = len(imgmx)
	h = len(imgmx[0])
	for y in range(0, h):
		for x in range(0, w):
			v = imgmx[x][y]
			if v > 200:
				print("@", end="")
			elif v > 150:
				print("%", end="")
			elif v > 100:
				print("#", end="")
			elif v > 50:
				print("*", end="")
			elif v > 35:
				print("=", end="")
			elif v > 25:
				print("-", end="")
			elif v > 15:
				print(":", end="")
			elif v > 5:
				print(".", end="")
			else:
				print(" ", end="")
			
		print()


def print_layer(prefix, layer):
	print(prefix, "\t", "name:", layer.get_config().get("name"))
	
	if layer.__class__.__name__ == "InputLayer":
		print(prefix, "\t", "batch_input_shape:", layer.get_config().get("batch_input_shape"))
	elif layer.__class__.__name__ == "Conv2D":
		print(prefix, "\t", "filters:", layer.get_config().get("filters"))
		print(prefix, "\t", "kernel_size:", layer.get_config().get("kernel_size"))
		print(prefix, "\t", "activation:", layer.get_config().get("activation"))
	elif layer.__class__.__name__ == "MaxPooling2D":
		print(prefix, "\t", "pool_size:", layer.get_config().get("pool_size"))
	elif layer.__class__.__name__ == "Dropout":
		print(prefix, "\t", "rate:", layer.get_config().get("rate"))
	elif layer.__class__.__name__ == "Dense":
		print(prefix, "\t", "units:", layer.get_config().get("units"))
		print(prefix, "\t", "activation:", layer.get_config().get("activation"))
	elif layer.__class__.__name__ == "Lambda":
		#print(prefix, "\t", "function:", layer.get_config().get("function"))
		print(prefix, "\t", "function:")
	
	#print(prefix, "\t", layer.get_config())
	print(prefix)





def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    
    seq.add(Conv2D(8, (3, 3), input_shape=(input_dimx,input_dimy,1), activation='relu'))
    seq.add(Conv2D(16, (3, 3), activation='relu'))
    
    #seq.add(Conv2D(16, (3, 3), input_shape=(input_dimx,input_dimy,1), activation='relu'))
    #seq.add(Conv2D(32, (3, 3), activation='relu'))
    #seq.add(MaxPooling2D(pool_size=(2,2)))
    #seq.add(Conv2D(32, (3, 3), activation='relu'))
    #seq.add(Conv2D(64, (3, 3), activation='relu'))
    #seq.add(MaxPooling2D(pool_size=(2,2)))
    
    seq.add(Dropout(0.1))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def make_network():
	
	# network definition
	base_network = create_base_network()
	
	input_a = Input(shape=(input_dimx,input_dimy,1))
	input_b = Input(shape=(input_dimx,input_dimy,1))
	
	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	
	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	
	model = Model([input_a, input_b], distance)
	return model


def load_weights (model, use_existing_model, model_optional):
	# load model if existing
	#model_file = Path("../../" + get_datadir ("../config.txt") + "/12-follow-model/model-1.h5")
	model_file = Path("model-1.h5")
	if use_existing_model and model_file.exists():
		print("> loading existing model file")
		#model.load_weights("../../" + get_datadir ("../config.txt") + "/12-follow-model/model-1.h5")
		model.load_weights("model-1.h5")
	else:
		print("> model file not existing or use_existing_model == False...")
		if (not model_optional):
			print ("model not optional for predict. quit now...")
			exit (1)
	return model


def compile_network (model):
	# compile network
	opt = RMSprop(lr = lr)
	model.compile(loss=contrastive_loss, optimizer=opt)
	#print(opt.get_config())
	return model






def dump_network (model):
	# dump network architecture
	for i in range(0,len(model.layers)):
		layer = model.get_layer(index = i)
		print("class:", layer.__class__.__name__)
		if layer.__class__.__name__ == "Sequential":
			for i in range(0,len(layer.layers)):
				sublayer = layer.get_layer(index = i)
				print("\t", "class:", sublayer.__class__.__name__)
				print_layer("\t", sublayer)
		else:
			print_layer("\t", layer)


def read_base_good (filename):
	# read text file and return its lines
	with open(filename) as f:
		content = f.readlines()
	
	# trim lines
	for i in range (len (content)):
		content [i] = content [i].strip ().split ("\t")
	
	# return list of list with base and good (bad could be added in a next step)
	return content
	





