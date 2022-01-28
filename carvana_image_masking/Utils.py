import tensorflow as tf
from tensorflow import keras

def save_model(model, path):
	model.save(path)

def load_model(path):
	return keras.models.load_model(path)

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