import tensorflow as tf
from tensorflow import keras

def save_model(model, path):
	model.save(path)

def load_model(path):
	return keras.models.load_model(path)