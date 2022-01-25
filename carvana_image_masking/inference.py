import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Utils import save_model, load_model
from data_loading import load_image

import os

def prepare_data(test_image_dir):

	image_list = os.listdir(test_image_dir)
	print("no. of test images: ", len(image_list))

	image_list.sort()

	test_path = []
	for i in image_list:
		path = os.path.join(test_image_dir, i)
		test_path.append(path)

	return test_path

test_image_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/test"
model_path = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "my_model")

test_image_list = prepare_data(test_image_dir)

sample_image = load_image(test_image_list[5]).reshape((1,256,256,3))

model = load_model(model_path)

pred = model.predict(sample_image)

print("predicted successfully")

plt.imshow(load_image(test_image_list[5]))
plt.show()
#plt.imshow(sample_test_mask)
#plt.show()
plt.imshow(pred[0,:,:,0])
plt.show()
