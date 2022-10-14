"""
Compare all nucl (in nucl_path) from one frame with all nucl from next frame

"""

from shutil import copyfile
import os.path
from PIL import Image, ImageDraw, ImageFont
from random import randint
import string, random
import numpy as np
import folnet
import os
import sys
import time
#import threading
from keras.models import clone_model
from keras.optimizers import RMSprop
import copy
from timeit import default_timer as timer
from keras import backend as K
import tensorflow as tf


if len(sys.argv) != 4:
	print("invalid number of arguments")
	print(f"usage: {sys.argv[0]} <nucl_path> <image_path> <output_path>")
	exit (1)

nucl_path = sys.argv [1]
image_path = sys.argv [2]
output_path = sys.argv [3]
#nucl_path = "../res/"
#image_path = "../image/"
#output_path = "../followed/"


colors23 = ("#48c9b0","#45b39d","#58d68d","#52be80","#5dade2","#5499c7","#af7ac5","#a569bd","#5d6d7e","#566573","#f4d03f","#f5b041","#eb984e","#d35400","#dc7633","#d35400","#a04000","#ec7063","#cd6155","#f0f3f4","#cacfd2","#aab7b8","#99a3a4")


def makepairs (filename, nextfilename):
	allpairs = []
	thismx = folnet.loadnucl (filename)
	nextmx = folnet.loadnucl (nextfilename)
	pair = np.asarray ([thismx, nextmx])
	allpairs.append (pair)
	allpairs = np.expand_dims (allpairs, 4)
	return allpairs





def gencolor ():
	x = random.randint (0, len (colors23) - 1)
	return colors23 [x]
	




def genstring ():
	chars = "0123456789ABCDEF"
	return "".join (random.choice (chars) for i in range (4))





def dodraw (nucl_path, nuclfilename, n, c, a):
	# get coords of nucl
	nuclx = int (nuclfilename [-14:-10])
	nucly = int (nuclfilename [-8:-4])
	
	# get size of nucleus
	nucl = Image.open (nucl_path + nuclfilename)
	nuclw, nuclh = nucl.size
	
	#print ("\t", nuclfilename, "\t", n, "\t", c, "\tage:", a, "\t", nuclx, "\t", nucly, "\t", nuclw, "\t", nuclh)
	
	# draw box around nucleus
	draw.line ((nuclx, nucly, nuclx + nuclw, nucly), fill=c)
	draw.line ((nuclx + nuclw, nucly, nuclx + nuclw, nucly + nuclh), fill=c)
	draw.line ((nuclx + nuclw, nucly + nuclh, nuclx, nucly + nuclh), fill=c)
	draw.line ((nuclx, nucly + nuclh, nuclx, nucly), fill=c)
	
	# write nucleus name
	totname = n + " [" + str (a) + "]"
	draw.text ((nuclx + 2, nucly), totname, fill=c, font=fnt)





class Job ():
	def __init__ (self, frame, nextframe, filename, nextfilename):
		self.frame = frame
		self.nextframe = nextframe
		self.filename = filename
		self.nextfilename = nextfilename
	
	
	def val (self):
		return self.frame, self.nextframe, self.filename, self.nextfilename
	
	
	def __str__ (self):
		return "[frame = " + self.frame + " nextframe = " + self.nextframe + " filename = " + self.filename + " nextfilename = " + self.nextfilename + "]"





class JobGenerator ():
	def __init__ (self, frames, allfiles):
		self.frames = frames
		self.allfiles = allfiles
		self.ptr_frame = 0
		self.ptr_filename = 0
		self.ptr_nextfilename = 0
		
		# make subsets of allfiles
		self.framefile = []
		for frame in self.frames:
			fstr = frame + '-ID'
			sub = [k for k in allfiles if fstr in k]
			self.framefile.append (sub)
		
		self.done = False
		
	
	
	def getJob (self):
		
		if self.done:
			return None
		
		subset = self.framefile [self.ptr_frame]
		nextsubset = self.framefile [self.ptr_frame + 1]
		
		filename = subset [self.ptr_filename]
		
		###
		# it is safe to ignore exceptions in wrkr-threads here
		###
		
		if self.ptr_nextfilename >= len(nextsubset):
			return None;
		
		
		nextfilename = nextsubset [self.ptr_nextfilename]
		
		job = Job (filename [:4], nextfilename [:4], filename, nextfilename)
		#print ("job =", job)
		
		self.ptr_nextfilename += 1
		
		self.nextindex = self.ptr_frame + 1
		if self.ptr_nextfilename == len (nextsubset):
			self.ptr_nextfilename = 0
			self.ptr_filename += 1
		
		if self.ptr_filename == len (subset):
			self.ptr_nextfilename = 0
			self.ptr_filename = 0
			self.ptr_frame += 1
			
		if self.ptr_frame == len (self.framefile) - 1:
			self.done = True
			
		return job




print ("nucl_path =", nucl_path)


# get font
#fnt = ImageFont.truetype ("UbuntuMono-R.ttf", 18)
fnt = ImageFont.truetype ("UbuntuMono-R.ttf", 32)


# collect frame numbers from nucl filenames as strings in 'frames' list
allfiles = [f for f in os.listdir (nucl_path) if os.path.isfile (os.path.join (nucl_path, f))]
allfiles.sort ()


frames = []  # prep empty list for frame numbers
for filename in allfiles:
	
	frm = filename [:filename.find ("-")]  # remove postfix
	
	# store frm in frames array if not yet existing
	if not frm in frames:
		frames.append (frm)

frames.sort ()

# max distance for two nuclei to be stored in cand list
threshold = 0.5


#dlogfile = open ("../../" + folnet.get_datadir ("../config.txt") + "/distance.log", "w")
dlogfile = open ("distance.log", "w")
dlogfile.write ("---------------------------------------------------------------------------------------\n")



jg = JobGenerator (frames, allfiles)

# Make dict ('cand') where index is combination of frame&id of one nucleus ('idthis') and frame&id of that same nucleus in the next frame ('idnext')
cand = {}  # prep dict for follow-up candidates



# prep network
model_bp = folnet.make_network ()
model_bp = folnet.load_weights (model_bp, True, False)
model_bp = folnet.compile_network (model_bp)
model_bp._make_predict_function ()
layer0 = model_bp.get_layer (index = 0)
#print ("layer0 input_shape =")
#print (layer0.input_shape)


model = clone_model (model_bp)
model.build (layer0.input_shape)
model.compile (loss=folnet.contrastive_loss, optimizer=RMSprop (lr = 0.000005))
model.set_weights (model_bp.get_weights ())
model._make_predict_function ()
begintime = timer ()

################################################################################
threadID = "workerx"
counter = 0
loopisdone = False;
while not loopisdone:
	#time_start = timer ()
	job = jg.getJob ()
	if job is None:
		print (threadID, "finished after", counter, "jobs")
		loopisdone = True
	
	if not loopisdone:
		counter += 1
		#print (self.name, "working on", job)
		frame = job.val () [0]
		nextframe = job.val () [1]
		filename = job.val () [2]
		nextfilename = job.val () [3]
		
		idthis = filename [-25:-16]  # combi of framenumber and ID of nucleus
		idnext = nextfilename [-25:-16]
		
		# convert filename and nextfilename into pair usable by network
		allpairs = makepairs (nucl_path + filename, nucl_path + nextfilename)  # expensive: 45 ms
		
		# feed images to siamese network
		pred = model.predict ([allpairs [:, 0], allpairs [:, 1]], verbose=0)  # expensive: 125 ms
		
		print (threadID, "feeding [", frame, "]", idthis, "and [", nextframe, "]", idnext, "-> distance is", pred [0][0])
		
		dlogfile.write (idthis + "\t" + idnext + "\t" + str (pred [0][0]) + "\n")
		
		# store interesting pairs in candidates (and forget really bad pairs)
		if (pred [0][0] < threshold):
			idnext = nextfilename [-25:-16]
			
			# store followup
			cand [(idthis, idnext)] = (pred [0][0], filename, nextfilename)
	


################################################################################
dlogfile.close ()


endtime = timer ()
dur = int ((endtime - begintime) * 1000)
print ("DURATION =", dur)


# 
print ("-" * 78)
followup = {}
for key in cand.keys ():
	
	# check if left and right side of key are both unique
	dist = cand [key][0]
	#print ("# KEY =", key, "DIST=", dist)
	left = key [0]
	right = key [1]
	hits_left = 0
	hits_right = 0
	for key2 in cand.keys ():
		left2 = key2 [0]
		right2 = key2 [1]
		if (left == left2):
			hits_left += 1
		
		if (right == right2):
			hits_right += 1
	
	if (hits_left == 1 and hits_right == 1):
		followup [left] = right
	else:
		# check if key exists with equal left part and smaller dist
		dist_min = sys.maxsize
		key_min = None
		for key2 in cand.keys ():
			left2 = key2 [0]
			if (left == left2):
				if (cand [key2][0] < dist):
					dist_min = cand [key2][0]
					key_min = key2
		
		if (dist <= dist_min):
			# check if key exists with equal right part and smaller dist
			dist_min = sys.maxsize
			key_min = None
			for key2 in cand.keys ():
				right2 = key2 [1]
				if (right == right2):
					if (cand [key2][0] < dist):
						dist_min = cand [key2][0]
						key_min = key2
			
			if (dist <= dist_min):
				followup [left] = right

# dump followup

# prep color and nuclname dict
name = {}
age = {}
color = {}

#logfile = open ("../../" + folnet.get_datadir ("../config.txt") + "/follow.log", "w")
logfile = open ("follow.log", "w")
logfile.write ("---------------------------------------------------------------------------------------\n")

print ("followup dump:")
for key in followup:
	#print (key, "->", followup [key], end="")
	
	# if key is also a value, then take color from followup where key is value
	if (key in followup.values ()):
		print ("[flw]", end="")
		logfile.write ("[flw]\t")
		
		# in followup find key where value is 'key'
		for item in followup.values ():
			if (item == key):
				# lookup key where value is 'key'
				for kk in followup.keys ():
					if (followup [kk] == key):
						lookfor = kk
						
						if kk in color:
							# normal situation
							name [item] = name [kk]
							age [item] = age [kk] + 1
							color [item] = color [kk]
						else:
							# weird situation (eg when to different movies are handles in one go)
							#name [key] = " " + genstring () + "X"
							name [key] = genstring () + "X"
							age [key] = 1
							color [key] = gencolor ()
	
	# else choose new random color
	else:
		print ("[new]", end="")
		logfile.write ("[new]\t")
		name [key] = genstring ()
		age [key] = 1
		color [key] = gencolor ()
	
	
	# print color if known
	if (key in color):
		print (" name =", name [key], end="")
		print (" age =", age [key], end="")
		#print (" color =", color [key], end="")
		logfile.write (name [key])
		logfile.write ("\t")
		logfile.write (str(age [key]))
	
	filename_left = cand [(key, followup [key])][1]
	filename_right = cand [(key, followup [key])][2]
	
	print ("\t", filename_left, "\t", filename_right, end="")
	logfile.write ("\t")
	logfile.write (filename_left)
	logfile.write ("\t")
	logfile.write (filename_right)
	logfile.write ("\n")
	
	print ()

#print ("color dump:")
#for key in color:
#	print (key, "->", color [key])
logfile.close ()

# loop through all frames
allframes = [f for f in os.listdir (image_path) if os.path.isfile (os.path.join (image_path, f))]
print ("[drawing] followed nucl in frames...")
for filename in allframes:
	#print ("filename =", filename)
	# read frame image
	im = Image.open (image_path + filename)
	
	# Convert grayscale to rgb
	rgb = Image.new ("RGBA", im.size)
	rgb.paste (im)
	im = rgb
	
	draw = ImageDraw.Draw (im)
	
	#frame = filename [-8:-4]
	frame = os.path.splitext (filename) [0]
	frame = str (frame).zfill (4)
	
	#print ("frame =", frame)
	for key in followup:
		#print ("# KEY =", key)
		keyframe = key[:4]
		#print ("keyframe =", keyframe)
		
		if (keyframe == frame):
			nuclfilename = cand [(key, followup [key])][1]
			#print ("[-] dodraw:", nuclfilename, "key =", key)
			dodraw (nucl_path, nuclfilename, name [key], color [key], age [key])
	
	for key in followup:
		
		# check if followup [key] exists as key itself
		keyexists = False
		for skey in followup:
			if (skey == followup [key]):
				keyexists = True
		
		if (not keyexists):
			#print ("VALUE NOT EXISTING AS KEY:", followup [key])
			
			keyframe = followup [key][:4]
			if (keyframe == frame):
				nuclfilename = cand [(key, followup [key])][2]
				#print ("[#] dodraw:", nuclfilename, "key =", key)
				dodraw (nucl_path, nuclfilename, name [key], color [key], age [key] + 1)
			
	
	
	# write frame with follwed nuclei
	del draw
	print ("do_predict :: im.save output_path =", output_path, "filename =", filename)
	im.save (output_path + filename, quality=100)



