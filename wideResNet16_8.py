import keras
import argparse
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, Flatten
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import pickle 

# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008


# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=64, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=2, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
                help='dataset. (default: cifar10)')
parser.add_argument('-k','--width', type=int, default=8, metavar='NUMBER',
                help='width(default: 8)')
    
args = parser.parse_args()

    
    
stack_n            = args.stack_n
layers             = 6 * stack_n + 4
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 64
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4
k                  = args.width


def residual_network(img_input,classes_num=10,stack_n=9):
    
    def residual_block(x,o_filters,stride=(1,1),increase=False):
        #stride = (1,1)
        #if increase:
           # stride = (2,2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=stride,padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    
# build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x128
    x = residual_block(x,128,stride=(1,1), increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x,128,stride=(1,1),increase=False)

    # input: 32x32x128 output: 16x16x256
    x = residual_block(x,256,stride=(2,2),increase=True)
    for _ in range(1, stack_n):
        x = residual_block(x,256,stride=(1,1),increase=False)
    
    # input: 16x16x256 output: 8x8x512
    x = residual_block(x,512,stride=(2,2),increase=True)
    for _ in range(1,stack_n):
        x = residual_block(x,512,stride=(1,1),increase=False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
   
    return x



if __name__ == '__main__':

    
    print("========================================") 
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+4)) 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    ## print("DATASET: {:}".format(args.dataset))
    print("== LOADING DATA... ==")


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
 
    #z-score normalization
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output) 
    print(resnet.summary())
    
    
    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist=resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         verbose=1,
                         callbacks=[LearningRateScheduler(scheduler)],
                         validation_data=(x_test, y_test))
    resnet.save('Wide_resnet_{:d}_{:d}.h5'.format(layers, k))
    
    #testing
    scores = resnet.evaluate(x_test, y_test, batch_size=128, verbose=2)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    # save results to pickle
    res={"history" : hist.history, "scores" : scores}
    pickle_out = open("ResNet{:d}_{:d}.pickle".format(layers, k),"wb")
    pickle.dump(res, pickle_out)
    pickle_out.close()

