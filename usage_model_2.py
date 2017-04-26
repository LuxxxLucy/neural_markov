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
from model2_pipeline import *
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

def get_index(x,x_s):
    for c,it in enumerate(x_s):
        if x==it:
            return c
        else:
            continue
    return 0

def conditiaonal_probability(target_words,sentence,keywords,model):
    sentence = preprocess_on_review_content("".join(sentence))
    word_map,vectors=load_meta_model()
    DEFAULT_MISSING_INDEX=str(len(word_map)+1)
    vectors[DEFAULT_MISSING_INDEX]=np.zeros(VECTOR_DIMENSION,dtype=float)
    X = list()
    vectors_of_words = [vectors[word_map[it]] for it in sentence ]
    x1 = np.zeros( (len(vectors_of_words),VECTOR_DIMENSION), dtype=float)
    for count,word in enumerate(vectors_of_words):
        x1[count]+=np.asarray(word,dtype=float)

    x2_temp = [ np.array(vectors[word_mapping(it,word_map)], dtype=float) for it in keywords[count:0:-1] ][:CONTEXT_WINDOW_SIZE]
    x2 = np.zeros( (CONTEXT_WINDOW_SIZE,VECTOR_DIMENSION),dtype=float)
    for c,it in enumerate(x2_temp):
        x2[c]+=x2_temp[c]

    for target_word in target_words:
        try:
            x3 = np.array( vectors[word_mapping(target_word,word_map)] )
            x3 = np.array(x3)
            c= get_index(target_word,sentence)
            x1_before,x1_current,x1_after = padding_zeros(x1,index=c,window_size=5,zero_shape=(100))
            X.append(np.vstack((x2,x1_before,x1_current,x1_after,x3)))
        except:
            print(c)
            print(x2.shape)
            print(x1_after.shape)
            print(x1_current.shape)
            print(x1_before.shape)
            print(x3.shape)

    X=np.array(X)
    # X,Y=construct_train_data_raw([sentence],[keys])
    X_o,X_s,X_current = construct_input_data(X)

    predictions=model.predict({'state_context_input': X_s, 'observation_context_input': X_o,'current_input':X_current},
        batch_size=32,verbose=2)
    return predictions


if __name__ == "__main__":

    word_map,vectors=load_meta_model()
    index_to_word= { item:key for key,item in word_map.items()}

    records=load_labeled_data("./data/tmp.json")


    model=load_model("./model/dis_enc_model.model")
    for i in range(10):
        record=records[i]
        sentence = record["content"]
        print("raw sentence is ")
        print("".join(sentence))
        real_keys=record["key"]
        print("real keywords as")
        print(real_keys)
        words = sentence
        pro=conditiaonal_probability(words,sentence,[],model)
        pr(pro)


    quit()
