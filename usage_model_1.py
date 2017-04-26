#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os,random
import numpy as np
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
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
from keras.models import Model,load_model
from keras.utils import np_utils
from tqdm import tqdm

from preprocessing import *
from model1_pipeline import *
from pprint import pprint as pr
import pickle

CONTEXT_WINDOW_SIZE = 3
VECTOR_DIMENSION=100
MAX_LENGTH=100

def extract_key_words(sentence):
    words = preprocess_on_review_content(sentence)
    keys=["不错"]
    class_number=find_class_number()
    while len(keys)<=2:
        new_keyword = get_new_keyword(words,keys)
        keys.append( new_keyword )

    return keys

def get_new_keyword(words,keys):
    # construct words and keys as input vectos
    DEFAULT_MISSING_INDEX=str(len(word_map)+1)
    vectors[DEFAULT_MISSING_INDEX]=np.zeros(VECTOR_DIMENSION,dtype=float)

    X = list()
    Y = list()
    vectors_of_words = [ vectors[word_map[it]] for it in words ]
    x1 = np.zeros( (len(vectors_of_words),VECTOR_DIMENSION), dtype=float)
    for count,word in enumerate(vectors_of_words):
        x1[count]+=np.asarray(word,dtype=float)

    for count,target_word in enumerate(keys):
        x2_temp = [ np.array(vectors[word_mapping(it,word_map)], dtype=float) for it in keys[count:0:-1] ][:CONTEXT_WINDOW_SIZE]
        x2 = np.zeros( (CONTEXT_WINDOW_SIZE,VECTOR_DIMENSION),dtype=float)
        for c,it in enumerate(x2_temp):
            x2[c]+=x2_temp[c]
        target_index=int(word_mapping(target_word,word_map))
        X.append((x2,x1 ))

    if(len(keys)==0):
        x2 = np.zeros( (CONTEXT_WINDOW_SIZE,VECTOR_DIMENSION),dtype=float)
        X.append( (x2,x1))

    X=np.array(X)

    X_o,X_s = construct_input_data(X)

    result = model.predict({'state_context_input': X_s, 'observation_context_input': X_o},batch_size=32,verbose=1)
    # print(result)
    t=result.argmax(1)
    s = index_to_word[str(t[0])]
    return s
    #
    # print( word_map["工作人员"])
    # print(t)

    # print(s)
    #
    # s = index_to_word[str(t[0]+1)]
    # print(s)
    # s = index_to_word[str(t[0]-1)]
    # print(s)
    #
    #
    # result = model.predict_classes({'state_context_input': X_s, 'observation_context_input': X_o},batch_size=32,verbose=1)
    # print(result)
    #
    # result = model.predict_proba({'state_context_input': X_s, 'observation_context_input': X_o},batch_size=32,verbose=1)
    # print(result)


def conditiaonal_probability(sentence,keys,model):

    model.predict()

    return

def construct_input_data(X):
    X_o = np.array([  np.vstack(np.asarray(it))  for it in X[:,1] ])
    X_s =  np.array([  np.concatenate(np.array(it))  for it in X[:,0] ])
    return X_o,X_s

if __name__ == "__main__":


    word_map,vectors=load_meta_model()
    index_to_word= { item:key for key,item in word_map.items()}


    records=load_labeled_data("./data/tmp.json")

    records=preprocessing.load_labeled_data("./data/tmp.json")
    temp=  [re["key"] for re in records]

    pr(temp[:10])
    class_number=find_class_number()
    X,Y=construct_train_data(records)
    total_number=len(X)
    Y = keras.utils.to_categorical(Y, num_classes=class_number+1)

    X_train = X[:10,:]
    Y_train = Y[:10]
    X_test = X[int(total_number*0.9):,:]
    Y_test = Y[int(total_number*0.9):]

    X_o = np.array([  np.vstack(np.array(it))  for it in X_train[:,0] ])
    print(X_o.shape)
    X_s =  np.array([  np.concatenate(np.array(it))  for it in X_train[:,0] ])
    print(X_s.shape)

    model=load_model("./model/lstm_enc_model.model")
    result = model.predict({'state_context_input': X_s, 'observation_context_input': X_o},batch_size=32,verbose=1)
    y= [ index_to_word[str(np.argmax(it))] for it in result ]
    y_ = [ index_to_word[str( np.argmax(it)  )] for it in Y_train]
    pr(y)
    pr(y_)


    score = model.evaluate({'state_context_input': X_s, 'observation_context_input': X_o},
          {'predictions':Y_train }, batch_size=128)
    pr(score)
    quit()


    for i in range(10):
        record=records[i]
        sentence = "".join(record["content"])
        standard_keys = record["key"]
        print("raw sentence is ")
        pr(sentence)
        print("standard key word is ")
        pr(standard_keys)



        keywords=extract_key_words(sentence)
        print("extracted key words are")
        pr(keywords)

    quit()



    class_number=find_class_number()
    X,Y=construct_train_data(records)
    total_number=len(X)
    Y = keras.utils.to_categorical(Y, num_classes=class_number+1)
    X_train = X[:int(total_number*0.9),:]
    Y_train = Y[:int(total_number*0.9)]
    X_test = X[int(total_number*0.9):,:]
    Y_test = Y[int(total_number*0.9):]

    X_o_test,X_s_test = construct_input_data(X_test)

    score = model.evaluate({'state_context_input': X_s_test, 'observation_context_input': X_o_test},
          {'predictions':Y_test }, batch_size=128)
    pr(score)



    conditiaonal_probability(setence,previous_key,model)
