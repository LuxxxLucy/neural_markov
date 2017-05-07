#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import re
from pprint import pprint as pr
from collections import defaultdict
import jieba
import jieba.posseg
import jieba.analyse
import gensim

import pickle

VECTOR_DIMENSION=100

def load_raw_data(filename):
    f=open(filename,"r")
    records = json.load(f)
    content=list()

    for line in records:
        id= line["product_id"]
        name= line["product_name"]
        score= line["score"]
        review_content= [it for it in jieba.cut(line["content"])]
        review_content=custom_strip(review_content)
        content.append( { "product_id":id,"product_name":name,"score":score,"content":review_content } )

    return content

def preprocess_on_review_content(sentence):
    sentence = [     it for it in jieba.cut(sentence)   ]
    return custom_strip(sentence)

def load_labeled_data(filename):
    f=open(filename,"r")
    records = json.load(f)

    content=list()

    for line in records:
        id= line["product_id"]
        name= line["product_name"]
        score= line["score"]
        review_content= line["content"]

        review_content=custom_strip(review_content)
        senti= line["sentiment"]
        words=line["key"]
        content.append( {"id":id,"name":name,"score":score,"senti":senti,"key":words,"content":review_content} )
        # key_word_extract(review_content)
    return content

def map_to_range(i,start,end,target_start,target_end):
    result = (i-start)/(end-start) * (target_end-target_start) + target_start
    return int(result)

def make_fake_labels(records):
    for rec in records:
        rec["key"]=key_word_extract( "".join(rec["content"]))
        rec["sentiment"]= map_to_range(len(rec["content"]),2,50,-2,2)

    return records


def key_word_extract(s):
    # for x, w in jieba.analyse.textrank(s, withWeight=True):
    #     print('%s %s' % (x, w))
    # for x, w in jieba.analyse.extract_tags(s, withWeight=True):
    #     print('%s %s' % (x, w))
    return jieba.analyse.textrank(s,withWeight=False)[:10]

def detect_and_delete(records):
    result=list()
    for r in records:
        if(spam_or_not(r["content"]) and len(r["key"])!=0 ):
            result.append(r)

    return result

def spam_or_not(sentence):
    if(len(sentence)<=1):
        return False
    else:
        return True

def custom_strip(sentence):
    spam_array=[" ","&","hellip",";","(",")","（","）"]
    result=[]
    for it in sentence:
        if(it in spam_array):
            continue
        pattern= re.compile("([a-zA-Z]+|[0-9]+)")
        if(pattern.match(it)!=None):
            continue
        result.append(it)

    return result

def save_as_file(records,filename="./data/tmp.json"):

    f=open(filename,"w",encoding="utf8")
    json.dump(records,f,ensure_ascii=False)
    f.close()
    return

def train_w2v(sentences):
    word_set = set([  it for re in sentences for it in re] )
    word_map = {word:str(c) for c,word in enumerate(word_set)}
    sentences= [   [ word_map[word] for word in sentence] for sentence in sentences]

    pr(word_map)
    # pr(sentences)
    voca_size=len(word_map)
    pr(voca_size)
    model = gensim.models.Word2Vec(sentences, min_count=1,size=VECTOR_DIMENSION)
    # model.save('/tmp/mymodel')
    # new_model = gensim.models.Word2Vec.load('/tmp/mymodel')


    # model=model.wv
    # mapping from word to index

    index_to_vector=dict()
    f_write=open("./model/vectors_txt_format.txt","w")
    for i in range(voca_size):
        if str(i) in model:
            f_write.write(str(i)+":"+" ".join([str(temp) for temp in model[str(i)]])+"\n")
            index_to_vector[str(i)]=[str(temp) for temp in model[str(i)]]
        else:
            print(str(i)+" is not in vocabulary")
    f_write.close()
    print(len(index_to_vector))


    # save two file
    # 1. mapping dictionary
    # 2. vectors corresponding to  each index
    with open("model/word_map.dict","wb") as f:
        pickle.dump(word_map,f)
    with open("model/vectors.dict","wb") as f:
        pickle.dump(index_to_vector,f)

    # in case of danger, save the model either
    with open("./model/genism_model_vector.test.db","wb") as f:
        model = gensim.models.Word2Vec(sentences,size=100,window=5, workers=4)
        pickle.dump(model,f)

    return

def process(query_id,data):
    total_com=len(data)
    result=dict()
    try:
        valid_com_num,key_com=delete_and_extract(data)
        status="ok"
        result["data"]=dict()
        result["data"]["commentsTotalNum"]=total_com
        result["data"]["validComNum"]=valid_com_num
        result["data"]["invalidComNum"]=total_com - valid_com_num
        result["data"]["validPercent"]=valid_com_num / total_com
        result["data"]["invalidPercent"]=invalid_com_num / total_com
        result["data"]["key_com"]=key_com
    except:
        status="error happened internal server error"

    result["query_id"]=query_id
    result["statue"]=status

    # now turns a dictionary into json string
    # result=json.dumps(result)

    return result

def delete_and_extract(records):
    content=list()

    for line in records:
        id= line["product_id"]
        name= line["product_name"]
        score= line["score"]
        review_content= [it for it in jieba.cut(line["content"])]
        review_content=custom_strip(review_content)
        content.append( { "product_id":id,"product_name":name,"score":score,"content":review_content } )
    valid_record = detect_and_delete(records)

    com_keys=key_word_extract( ".".join([ "".join(rec["content"]) for rec in valid_record ]) )
    return len(valid_record),com_keys


if __name__ == "__main__":
    # raw = load_raw_data("./data/jd_comment_items.json")
    # print("now is with label")
    # print(len(raw))
    # raw = make_fake_labels(raw)
    # raw=detect_and_delete(raw)
    #
    # save_as_file(raw)
    # pr(raw[0:5])
    # print(len(raw))
    #
    records=load_labeled_data("./data/tmp.json")
    pr(records[0:5])
    sentences=[ record["content"] for record in records]
    train_w2v(sentences)

    # label_data = load_labeled_data("./data/example.json")
