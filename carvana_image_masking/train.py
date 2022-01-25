import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data_loading import prepare_data, load_data, load_image
from model_Unet import Unet_model, Unet_model_V2
from Utils import save_model, load_model
import os

from datetime import datetime
from packaging import version

#from tensorflow.python.compiler.mlcompute import mlcompute

'''tf.compat.v1.disable_eager_execution()
mlcompute.set_mlc_device(device_name='gpu')
print("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
print("is_tf_compiled_with_apple_mlc %s" % mlcompute.is_tf_compiled_with_apple_mlc())
print(f"eagerly? {tf.executing_eagerly()}")
print(tf.config.list_logical_devices())'''

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

#print(model.summary())
model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

print("model loaded successfully..")

batchsize = 64

model_history = model.fit(x_train, y_train, epochs=10, batch_size=batchsize, validation_data=(x_test, y_test))
print("model training completed..")

#save the model

model_path = os.path.join("/Users/rajakumar/Desktop/DL_Learnings/kaggle/carvana_image_masking/", "my_model")
save_model(model, model_path)

#save logs





#loaded_model = load_model(model_path)
#print(loaded_model.summary())

#loaded_model_history = loaded_model.fit(x_train, y_train, epochs=2, batch_size=batchsize, validation_data=(x_test, y_test))


'''sample_test_img = load_image(image_path[25])
sample_test_mask = load_image(mask_path[25])

sample_test_img = sample_test_img.reshape((1,256,256,3))

pred = model.predict(sample_test_img)

print("predicted successfully")

plt.imshow(load_image(image_path[25]))
plt.show()
plt.imshow(sample_test_mask)
plt.show()
plt.imshow(pred[0,:,:,0])
plt.show()'''
