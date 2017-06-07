from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dense, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint
from scipy.misc import imresize, imsave
from keras.datasets import cifar10
import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
from skimage import color

x   = Input(shape=(32, 32, 1))
ca  = Conv2D(32, (3, 3), padding="same", kernel_initializer="glorot_normal")(x)
bna = BatchNormalization(axis=1)(ca)
ac0 = Activation("relu")(bna)
c1  = Conv2D(64, (3, 3), padding="same", kernel_initializer="glorot_normal")(ac0)
bn1 = BatchNormalization(axis=1)(c1)
ac1 = Activation("relu")(bn1)
p1  = MaxPooling2D((2, 2), padding="same")(ac1)
c2  = Conv2D(128, (3, 3), padding="same", kernel_initializer="glorot_normal")(p1)
bn2 = BatchNormalization(axis=1)(c2)
ac2 = Activation("relu")(bn2)
p2  = MaxPooling2D((2, 2), padding="same")(ac2)
c3  = Conv2D(256, (3, 3), padding="same", kernel_initializer="glorot_normal")(p2)
bn3 = BatchNormalization(axis=1)(c3)
ac3 = Activation("relu")(bn3)
c4  = Conv2D(256, (3, 3), padding="same", kernel_initializer="glorot_normal")(ac3)
bn4 = BatchNormalization(axis=1)(c4)
ac4 = Activation("relu")(bn4)
p3  = MaxPooling2D((2, 2), padding="same")(ac4)
dc3 = Conv2DTranspose(256, (3, 3), padding="same", kernel_initializer="glorot_normal")(p3)
bn9 = BatchNormalization(axis=1)(dc3)
ac9 = Activation("relu")(bn9)
dc4 = Conv2DTranspose(256, (3, 3), padding="same", kernel_initializer="glorot_normal")(ac9)
bn10 = BatchNormalization(axis=1)(dc4)
ac10 = Activation("relu")(bn10)
up2 = Conv2DTranspose(128, (2, 2), strides=(2,2), kernel_initializer="glorot_normal")(ac10)
bn11 = BatchNormalization(axis=1)(up2)
ac11 = Activation("relu")(bn11)
dc5 = Conv2DTranspose(64, (3, 3), padding="same", kernel_initializer="glorot_normal")(ac11)
bn12 = BatchNormalization(axis=1)(dc5)
ac12 = Activation("relu")(bn12)
up3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer="glorot_normal")(ac12)
bn13 = BatchNormalization(axis=1)(up3)
ac13 = Activation("relu")(bn13)
dc6 = Conv2DTranspose(16, (3, 3), padding="same", kernel_initializer="glorot_normal")(ac13)
bn14 = BatchNormalization(axis=1)(dc6)
ac14 = Activation("relu")(bn14)
up4 = Conv2DTranspose(2, (2, 2), strides=(2, 2), activation="sigmoid", kernel_initializer="glorot_normal")(ac14)
model = Model(inputs=x, outputs=up4)
print(model.summary())

def normalize_lab(x, input=0): # 0-x, 1-y #
	if input==0:
		x = x / 100.
	else:
		n = x.shape[0]
		x[:n, :32, :32, 0] = (x[:n, :32, :32, 0] + 86.185) / 184.439
		x[:n, :32, :32, 1] = (x[:n, :32, :32, 1] + 107.863) / 202.345
	return x

def unnormalize_lab(x, input=0):
	if input==0:
		x = x * 100.
	else:
		n = x.shape[0]
		x[:n, :32, :32, 0] = x[:n, :32, :32, 0] * 184.439 - 86.185
		x[:n, :32, :32, 1] = x[:n, :32, :32, 1] * 202.345 - 107.863
	return x

def process_imgs(x, n):
	aux = color.rgb2lab(x[:n])
	res_x = aux[:n, :32, :32, 0].reshape(n, 32, 32, 1)
	res_y = aux[:n, :32, :32, 1:]
	return res_x, res_y


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.random.shuffle(x_train)
np.random.shuffle(x_test)
for i in range(500):
	imsave("./Truths/truth"+str(i)+".jpg", x_test[i])
x_train, y_train = process_imgs(x_train, 50000)
x_test, y_test = process_imgs(x_test, 10000)
# Normalize train&test #
x_train = normalize_lab(x_train, 0)
y_train = normalize_lab(y_train, 1)
x_test  = normalize_lab(x_test, 0)
y_test = normalize_lab(y_test, 1)

# Load #
model.load_weights("best_colorization.h5")

model.compile(optimizer="adam", loss="mse", metrics=["mse"])

# Checkpoint #
filepath="best_colorization.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#model.fit(x_train, y_train, epochs=1000, batch_size=256, verbose=2, validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks_list)

# Save #
model_json = model.to_json()
with open("color_model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("color_model.h5")


for i in range(500):
	res = model.predict(np.array([x_test[i]]))
	l   = unnormalize_lab(np.array([x_test[i]]), 0)
	ab  = unnormalize_lab(res, 1)
	res = np.concatenate((l, ab), axis=3)
	res = color.lab2rgb(res[0])
	imsave("./Painteds/predicted"+str(i)+".jpg", res)

