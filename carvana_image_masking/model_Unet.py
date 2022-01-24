import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, concatenate, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model

def Unet_model(output_channels):
	base_model = tf.keras.applications.MobileNetV2(input_shape=[256,256,3], include_top=False)

	layer_names = ['block_1_expand_relu', 
	'block_3_expand_relu', 'block_6_expand_relu',
	'block_13_expand_relu', 'block_16_project']

	base_outputs = [base_model.get_layer(name).output for name in layer_names]
	down_sample = tf.keras.Model(inputs=base_model.input, outputs = base_outputs)

	down_sample.trainable = False

	up_sample = [pix2pix.upsample(512,3),
	pix2pix.upsample(256,3),
	pix2pix.upsample(128,3),
	pix2pix.upsample(64,3),]

	inputs = tf.keras.layers.Input(shape=(256,256,3))

	#downsample
	downsample_outputs = down_sample(inputs)
	x = downsample_outputs[-1]
	intermediate_output = list(reversed(downsample_outputs[:-1]))

	for i in range(len(up_sample)):
		x = up_sample[i](x)
		concat = tf.keras.layers.Concatenate()
		x = concat([x, intermediate_output[i]])

	last_layer = tf.keras.layers.Conv2DTranspose(filters = output_channels,
		kernel_size=3, strides=2, padding='same')
	x = last_layer(x)

	#final_layer = tf.keras.layers.Reshape((256,256))

	#X = final_layer(x)

	#x = x.reshape(256,256)
	final_layer = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')

	x = final_layer(x)

	return tf.keras.Model(inputs=inputs, outputs=x)

def Unet_model_V2(input_shape):
    input_layer = Input(shape = input_shape)
    
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(input_layer)
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
    conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
    conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
    conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
    conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    return Model(input_layer, conv10)


'''output_classes = 2

model = Unet_model(output_classes)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuray'])

#tf.keras.utils.plot_model(model,show_shapes=True)

print(model.summary())'''