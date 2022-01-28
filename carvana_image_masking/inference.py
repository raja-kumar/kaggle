import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Utils import save_model, load_model
from data_loading import load_image
import cv2
import os
import numpy as np
import pandas as pd


def prepare_data(test_image_dir):

	image_list = os.listdir(test_image_dir)
	print("no. of test images: ", len(image_list))

	image_list.sort()

	test_path = []
	for i in image_list:
		path = os.path.join(test_image_dir, i)
		test_path.append(path)

	return test_path

def encode_mask(pred_mask):
	arr = pred_mask.flatten()
	arr[0] = 0
	arr[-1] = 0

	ret_arr = np.where(arr[1:] != arr[:-1])[0] + 2
	ret_arr[1::2] = ret_arr[1::2] - ret_arr[:-1:2]

	return ret_arr


test_image_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/test"
model_path = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "my_model")
output_csv = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "submission.csv")

test_image_list = prepare_data(test_image_dir)

#sample_image = load_image(test_image_list[5]).reshape((1,256,256,3))

model = load_model(model_path)
j = 0
#df = pd.DataFrame(columns=['img', 'rle_mask'])
#print(df.head())

output_dict = {'img': [], 'rle_mask': []}
total = len(test_image_list)
for image_path in test_image_list:
	img_name = image_path.split('/')[-1]
	#print('image name: ', img_name)
	image = load_image(image_path).reshape((1,256,256,3))
	pred = model.predict(image)
	pred = 255. * cv2.resize(pred[0], (1918,1280))
	pred[pred <= 127] = 0
	pred[pred > 127] = 1
	pred_encoded = encode_mask(pred.astype(np.uint8))
	output_dict['img'].append(img_name)
	str_pred = ' '.join(str(x) for x in pred_encoded)
	output_dict['rle_mask'].append(str_pred)

	#plt.imshow(load_image(image_path))
	#plt.show()
	#plt.imshow(pred)
	#plt.show()
	print("predicted successfully", j, '/', total)
	j += 1

df = pd.DataFrame(output_dict)
df.set_index('img', inplace=True)
df.to_csv(output_csv)
#print("predicted successfully")

#pred = cv2.resize(pred[0], (1918,1280))
#plt.imshow(load_image(test_image_list[5]))
#plt.show()
#plt.imshow(sample_test_mask)
#plt.show()
#plt.imshow(pred[0,:,:,0])
#plt.imshow(pred)
#plt.show()
