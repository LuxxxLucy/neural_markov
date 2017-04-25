#!/usr/bin/env python
#
import os,random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from tensorflow.examples.tutorials.mnist import input_data
# from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm
import preprocessing
from pprint import pprint as pr


def build_model():
    # Build Generative model ...
    nch = 200
    g_input = Input(shape=[100])
    H = Dense(nch*14*14, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Reshape( [ 14, 14,nch] )(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    H = Reshape( [ 1,28,28] )(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()

    return generator


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def plot_loss(losses):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    # plt.figure(figsize=(10,8))
    # plt.plot(losses["d"], label='discriminitive loss')
    # plt.plot(losses["g"], label='generative loss')
    # plt.legend()
    # plt.show()

def show_sample_result():
    pass


def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):
        # real image
        data_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        # generative image
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((data_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        # one-hot code
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        d_loss  = discriminator.train_on_batch(X,y)

        # Updates plots
        # if e%plt_frq==plt_frq-1:
            # plot_loss(losses)
            # plot_gen()

if __name__ == "__main__":

    records=preprocessing.load_labeled_data("./data/tmp.json")
    pr(records[0:5])




    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train = mnist.train.images[:,:]
    y_train = mnist.train.labels[:,:]

    X_test = mnist.test.images[:,:]
    y_test = mnist.test.labels[:,:]

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize the pixel value
    # X_train /= 255
    # X_test /= 255


    print(np.min(X_train), np.max(X_train))
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


    shape = X_train.shape[1:]
    print(shape)
    dropout_rate = 0.25
    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-3)

    model=build_generative_model()

    ntrain = 10000
    trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
    XT = X_train[trainidx,:,:,:]

    # Pre-train the discriminator network ...
    noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
    generated_images = generator.predict(noise_gen)
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1

    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=1, batch_size=128)
    y_hat = discriminator.predict(X)

    # Measure accuracy of pre-trained discriminator network
    y_hat_idx = np.argmax(y_hat,axis=1)
    y_idx = np.argmax(y,axis=1)
    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print ("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))

    # Train for 6000 epochs at original learning rates
    train_for_n(nb_epoch=6000, plt_frq=500,BATCH_SIZE=32)

    # train for 2000 epochs at reduced learning rates
    opt.lr.set_value(1e-5)
    dopt.lr.set_value(1e-4)
    train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)

    # Train for 2000 epochs at reduced learning rates
    opt.lr.set_value(1e-6)
    dopt.lr.set_value(1e-5)
    train_for_n(nb_epoch=2000, plt_frq=500,BATCH_SIZE=32)
