import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.initializers import he_normal
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import pickle 
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    if epoch >125:
        lrate =0.0001
    return lrate

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
print("aaa")
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

# build model
weight_decay = 1e-4
model = Sequential()

# block 1

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal(),input_shape=x_train.shape[1:]))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# block 2

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# block 3

model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

# block 4

model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

# block 5

model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

# block 6

model.add(Flatten())
model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(Activation('selu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal()))
model.add(BatchNormalization())
model.add(Activation('softmax'))

print('model_built')


#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)
 
#training
batch_size = 64
 
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
hist=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=200,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])

#save to disk
model_json = model.to_json()
with open('model_vgg.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_vgg.h5') 
 
#testing
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=2)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

# save results to pickle
res={"history" : hist.history, "scores" : scores}
pickle_out = open("vgg_200epoch.pickle","wb")
pickle.dump(res, pickle_out)
pickle_out.close()
