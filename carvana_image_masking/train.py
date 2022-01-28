import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data_loading import prepare_data, load_data, load_image
from model_Unet import Unet_model, Unet_model_V2
from Utils import save_model, load_model
import os

from datetime import datetime
from packaging import version

root_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/"
train_image_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/train"
train_mask_dir = "/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/train_masks"

image_path, mask_path = prepare_data(train_image_dir, train_mask_dir)

x_train = load_data(image_path[:4600])
x_test = load_data(image_path[4600:])
y_train = load_data(mask_path[:4600])
y_test = load_data(mask_path[4600:])

print("data loaded..")

#to save logs
#logdir= os.path.join(root_dir, "logs/fit/", datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#define model
output_classes = 1
model = Unet_model(output_classes)
model_v2 = Unet_model_V2((256,256,3))

#compile and train model
model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
print("model loaded successfully..")

batchsize = 64

model_history = model.fit(x_train, y_train, epochs=10, batch_size=batchsize, validation_data=(x_test, y_test))
print("model training completed..")

#save the model
model_path = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "my_model")
save_model(model, model_path)
