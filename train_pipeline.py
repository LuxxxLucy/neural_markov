#!/usr/bin/env python
#
import os,random
import numpy as np
# import theano as th
# import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
# from tensorflow.examples.tutorials.mnist import input_data
# from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm
#


import preprocessing
from pprint import pprint as pr
import pickle

CONTEXT_WINDOW_SIZE = 3
VECTOR_DIMENSION=100
MAX_LENGTH=100

def build_model():
    # Build Generative model ...
    # nch = 200
    # g_input = Input(shape=[100])
    # H = Dense(nch*14*14, init='glorot_normal')(g_input)
    # H = BatchNormalization(mode=2)(H)
    # H = Activation('relu')(H)
    # H = Reshape( [ 14, 14,nch] )(H)
    # H = UpSampling2D(size=(2, 2))(H)
    # H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H)
    # H = Activation('relu')(H)
    # H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
    # H = BatchNormalization(mode=2)(H)
    # H = Activation('relu')(H)
    # H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
    # H = Reshape( [ 1,28,28] )(H)
    # g_V = Activation('sigmoid')(H)
    # generator = Model(g_input,g_V)
    # generator.compile(loss='binary_crossentropy', optimizer=opt)
    # generator.summary()

    # now start building network model

    input_1 = Input(shape=(3*VECTOR_DIMENSION,),name="state_context_input")
    encoded_y = Dense(VECTOR_DIMENSION,activation='relu')(input_1)

    input_2 = Input(shape=(None,VECTOR_DIMENSION),name="observation_context_input")
    encoded_x_y = LSTM(VECTOR_DIMENSION)(input_2)

    merged_vector = keras.layers.concatenate([encoded_x_y, encoded_y], axis=-1)
    predictions= Dense(class_number+1,activation='relu',name="predictions")(merged_vector)

    model = Model(inputs=[input_1,input_2],outputs=predictions)

    opt = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    model.summary()
    return model


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
    pass

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

def load_meta_model():
    word_map = pickle.load(open("./model/word_map.dict","rb"))
    vectors = pickle.load(open("./model/vectors.dict","rb"))
    return word_map,vectors

def word_mapping(word,word_map):
    try:
        return word_map[word]
    except:
        DEFAULT_MISSING_INDEX=str(len(word_map)+1)
        return DEFAULT_MISSING_INDEX

def construct_train_data(records):
    word_map,vectors=load_meta_model()
    DEFAULT_MISSING_INDEX=str(len(word_map)+1)
    vectors[DEFAULT_MISSING_INDEX]=np.zeros(VECTOR_DIMENSION,dtype=float)

    X = list()
    Y = list()
    for record in tqdm(records):
        vectors_of_words = [vectors[word_map[it]] for it in record["content"] ]

        x1 = np.zeros( (len(vectors_of_words),VECTOR_DIMENSION), dtype=float)
        for count,word in enumerate(vectors_of_words):
            x1[count]+=np.asarray(word,dtype=float)

        for count,target_word in enumerate(record["key"]):
            x2_temp = [ np.array(vectors[word_mapping(it,word_map)], dtype=float) for it in record["key"][count:0:-1] ][:CONTEXT_WINDOW_SIZE]
            x2 = np.zeros( (CONTEXT_WINDOW_SIZE,VECTOR_DIMENSION),dtype=float)
            for c,it in enumerate(x2_temp):
                x2[c]+=x2_temp[c]

            target_index=int(word_mapping(target_word,word_map))
            X.append((x2,x1 ))
            Y.append(target_index)


    print("data construct okay","train data size ",len(X),len(Y))


    return np.array(X),np.array(Y)

def find_class_number():
    word_map,vectors=load_meta_model()
    print("class number is ",len(word_map))
    return len(word_map)+1

if __name__ == "__main__":

    records=preprocessing.load_labeled_data("./data/tmp.json")
    class_number=find_class_number()
    X,Y=construct_train_data(records)
    total_number=len(X)
    Y = keras.utils.to_categorical(Y, num_classes=class_number+1)

    X_train = X[:int(total_number*0.9),:]
    Y_train = Y[:int(total_number*0.9)]
    X_test = X[int(total_number*0.9):,:]
    Y_test = Y[int(total_number*0.9):]



    X_o = np.array([  np.vstack(np.array(it))  for it in X_train[:,0] ])
    print(X_o.shape)
    pr(X_o[:2])
    X_s =  np.array([  np.concatenate(np.array(it))  for it in X_train[:,0] ])
    print(X_s.shape)
    pr(X_s[:2])



    model=build_model()
    model.fit({'state_context_input': X_s, 'observation_context_input': X_o},
          {'predictions':Y_train },
          epochs=50, batch_size=32)


    X_o_test= X_test[:,1]
    X_s_test= np.hstack(X_test[:,0])

    score = model.evaluate({'state_context_input': X_s_test, 'observation_context_input': X_o_test},
          {'predictions':Y_test }, batch_size=128)


    print(np.min(X_train), np.max(X_train))
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


    shape = X_train.shape[1:]
    print(shape)
    dropout_rate = 0.25
    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-3)


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
