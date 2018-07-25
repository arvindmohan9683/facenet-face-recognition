from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

PADDING = 50
ready_to_detect_identity = True

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):

	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

	# Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	# Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	# Step 3: subtract the two previous distances and add alpha.
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	# Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

	return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)



def prepare_database():
	database = {}

	# load all the images of individuals to recognize into the database
	for file in glob.glob("images/*"):
		identity = os.path.splitext(os.path.basename(file))[0]
		database[identity] = img_path_to_encoding(file, FRmodel)

	return database


def who_is_it(image, database, model):

	encoding = img_path_to_encoding(image, model)
	min_dist = 100
	identity = None

	# Loop over the database dictionary's names and encodings.
	for (name, db_enc) in database.items():

		# Compute L2 distance between the target "encoding" and the current "emb" from the database.
		dist = np.linalg.norm(db_enc - encoding)

		print('distance for %s is %s' %(name, dist))

		# If this distance is less than the min_dist, then set min_dist to dist, and identity to name
		if dist < min_dist:
			min_dist = dist
			identity = name

	if min_dist > 0.52:
		return None
	else:
		return str(identity)


database = prepare_database()
#create folder testimages and place images to be recognised in it
for file in glob.glob("testimages/*"):
	print("file name is "+file)
	answer = who_is_it(file,database,FRmodel)
	print (answer)
