import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Utils import save_model, load_model, prepare_data, encode_mask
from data_loading import load_image
import cv2
import os
import numpy as np
import pandas as pd

test_image_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/test"
model_path = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "my_model")
output_csv = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "submission.csv")

test_image_list = prepare_data(test_image_dir)

model = load_model(model_path)
j = 0

output_dict = {'img': [], 'rle_mask': []}
total = len(test_image_list)
for image_path in test_image_list:
	img_name = image_path.split('/')[-1]
	image = load_image(image_path).reshape((1,256,256,3))
	pred = model.predict(image)
	pred = 255. * cv2.resize(pred[0], (1918,1280))
	pred[pred <= 127] = 0
	pred[pred > 127] = 1
	pred_encoded = encode_mask(pred.astype(np.uint8))
	output_dict['img'].append(img_name)
	str_pred = ' '.join(str(x) for x in pred_encoded)
	output_dict['rle_mask'].append(str_pred)

	print("predicted successfully", j, '/', total)
	j += 1

df = pd.DataFrame(output_dict)
df.set_index('img', inplace=True)
df.to_csv(output_csv)