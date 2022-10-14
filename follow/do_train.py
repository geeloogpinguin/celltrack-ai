'''Train a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
import numpy as np
#np.random.seed(1337)  # for reproducibility

import folnet
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, Adadelta
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def unison_shuffled_copies (a, b):
	assert len (a) == len (b)
	p = np.random.permutation (len (a))
	#print ("p =", p)
	return a [p], b [p]


########################################################################################################################

# parameters
use_existing_model = False
#base_good_filename = "../../" + folnet.get_datadir ("../config.txt") + "/new_nucl_followup_list_good_only.txt"
#nucl_path = "../../" + folnet.get_datadir ("../config.txt") + "/11-predicted-nuclei/"
base_good_filename = "../new_nucl_followup_list_good_only.txt"
nucl_path = "../objseg/res/"

# make list of all nucl
all_nucl = [f for f in os.listdir (nucl_path) if os.path.isfile (os.path.join (nucl_path, f))]


# make list of base, good and bad nuclei
base_good_bad = folnet.read_base_good (base_good_filename)
for line in base_good_bad:
	# get frame number and id of good
	frame_good = line [1][-24:-21]
	id_good = line [1][-18:-16]
	
	# make list of nuclei where frame equals frame of good and id not equals id of good
	badlist = []
	for nucl in all_nucl:
		nucl_frame = nucl [-24:-21]
		nucl_id = nucl [-18:-16]
		
		if (nucl_frame == frame_good and nucl_id != id_good):
			badlist.append (nucl)
			
	
	# choose random nucleus from badlist
	line.append (badlist [random.randint (0, len (badlist) - 1)])


# make good and bad pairs
allpairs = []
allys = []
for line in base_good_bad:
	print (line)
	basemx = folnet.loadnucl (nucl_path + line [0])
	goodmx = folnet.loadnucl (nucl_path + line [1])
	badmx = folnet.loadnucl (nucl_path + line [2])
	goodpair = np.asarray ([basemx, goodmx])
	badpair = np.asarray ([basemx, badmx])
	allpairs.append (goodpair)
	allys.append (1)
	allpairs.append (badpair)
	allys.append (0)
	

# convert lists to nparrays
all_pairs = np.array (allpairs)
all_y = np.array (allys)


# shuffle all_pairs and all_y in same way
all_pairs, all_y = unison_shuffled_copies (all_pairs, all_y)


# add dimensions as required for unknown reason
all_pairs = np.expand_dims (all_pairs, 4)


# split pairs in tr and te
te_size = len (all_pairs) // 2  # size of evaluation set
tr_pairs = all_pairs [te_size:]
tr_y = all_y [te_size:]

te_pairs = all_pairs [:te_size]
te_y = all_y [:te_size]


# print stuff
print (len (all_pairs))
print (len (tr_pairs))
print (len (te_pairs))
print (tr_pairs.shape)
print (tr_y.shape)
print (te_pairs.shape)
print (te_y.shape)







# make network
model = folnet.make_network ()
model = folnet.load_weights (model, use_existing_model, True)
model = folnet.compile_network (model)
folnet.dump_network (model)


epochs = 2000000
for epoch in range (0, epochs):
	print("================= Epoch =", (epoch + 1), "/", epochs, "=================")
	
	model.fit ([tr_pairs [:, 0], tr_pairs [:, 1]], tr_y, validation_data=([te_pairs [:, 0], te_pairs [:, 1]], te_y), batch_size=128, epochs=1)
	# compute final accuracy on training and test sets
	pred = model.predict ([tr_pairs [:, 0], tr_pairs [:, 1]], verbose=1)
	tr_acc = folnet.compute_accuracy (pred, tr_y)
	pred = model.predict ([te_pairs [:, 0], te_pairs [:, 1]], verbose=1)
	te_acc = folnet.compute_accuracy (pred, te_y)
	
	#print (pred)
	
	print ('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	print ('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
	
	sc = 0
	for i in range(0, len (te_pairs)):
		if (pred [i] < 0.5 and te_y [i] == 1) or (pred [i] > 0.5 and te_y [i] == 0):
			sc += 1
		
	print ("* ev_score:", sc, "out of", len (te_pairs))
	
	# save model
	print ("> saving model...")
	#model.save_weights ("../../" + folnet.get_datadir ("../config.txt") + "/12-follow-model/model-1.h5")
	model.save_weights ("./model-1.h5")
	print ()








