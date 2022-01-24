import numpy as np
import pandas as pd
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

def prepare_data(train_image_dir, train_mask_dir):
	#train_image_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/train"
	#train_mask_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/train_masks"
	image_names = os.listdir(train_image_dir)
	mask_names = os.listdir(train_mask_dir)

	print("no. of training images: ", len(image_names))
	print("no. of training masks: ", len(mask_names))
	image_names.sort()
	mask_names.sort()

	image_path = []
	mask_path = []
	for i in image_names:
		path = os.path.join(train_image_dir, i)
		image_path.append(path)
	for j in mask_names:
		path = os.path.join(train_mask_dir, j)
		mask_path.append(path)

	return image_path, mask_path

def load_data(data_path):

	images = []

	for i in data_path:
		image = Image.open(i)
		#print('original shape: ', image.size)
		image = image.resize((256,256))
		image = np.array(image)
		images.append(image)

	images = np.array(images)
	print("done")

	return images

def load_image(image_name):
	image = Image.open(image_name)
	image = image.resize((256,256))
	image = np.array(image)

	return image

#x_train = load_data(image_path[:4001])
#x_test = load_data(image_path[4001:])
#y_train = load_data(mask_path[:4001])
#y_test = load_data(mask_path[4001:])

#print('input size: ', x_test.shape)
#print('output size: ', y_test.shape)

'''plt.imshow(x_train[0])
plt.show()
plt.imshow(y_train[0])
plt.show()'''