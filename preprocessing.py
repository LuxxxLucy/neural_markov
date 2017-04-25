import json
import re
from pprint import pprint as pr
from collections import defaultdict
import jieba
import jieba.posseg
import jieba.analyse
import genism

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
        content.append( {"id":id,"name":name,"score":score,"senti":senti,"key word":words,"content":review_content} )
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
    return jieba.analyse.textrank(s,withWeight=False)[:4]

def detect_and_delete(records):
    result=list()
    for r in records:
        if(spam_or_not(r["content"]) and len(r["key"])!=0 ):
            result.append(r)

    return result

def spam_or_not(sentence):
    '''
    if this sentence is a spam then return false otherwise true(which indicates it's normal
    '''
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

def train_w2v(sentence):
    model = gensim.models.Word2Vec(sentences, min_count=1,size=VECTOR_DIMENSION)
    # model.save('/tmp/mymodel')
    # new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

    model=model.wv
    # mapping from word to index
    word_map = {word:c for c,word in enumerate(model)}
    index_to_vector={ word_map[word] : model[word] for word in model}



    return


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
    records=preprocessing.load_labeled_data("./data/tmp.json")
    pr(records[0:5])
    sentences=[ record["content"] for record in raw]
    train_w2v(sentences)

    # label_data = load_labeled_data("./data/example.json")
