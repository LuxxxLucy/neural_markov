#encoding=utf-8

from flask import Flask, request
import os
from pprint import pprint as pr
import requests

from preprocessing import *

app = Flask(__name__)


@app.route("/main",methods=['POST'])
def main():
    # return app.send_static_file('base.html')
    # for it in request.files:
    #     f=request.files[it]
    #     f.save('./'+"temp_working.json")
    records=request.json
    print("got a sequence of data")
    pr(records)
    # records=json.load(open("./temp_working.json"))
    result= process(records)
    # try:
    pr(result)
    header={"content-type": "application/json"}
    r=requests.post("http://115.159.91.188:3000/statue",json=result,headers=header)
    # except:
    #     print("sending data back error")

    return json.dumps(result)


@app.route("/test",methods=['GET'])
def test():
    return send_test()

def send_test():

    result={
    "queryID": 1000,
    "statue": 'ok',
    "data":{
        "commentsTotalNum": 10000,
        "validComNum": 7500,
        "invalidComNum": 2500,
        "validPercent":75,
        "invalidPercent":25,
        "1starNum":4000,
        "2starsNum":2000,
        "3starsNum":1000,
        "4starsNum":500,
        "5starsNum":500,
        "avergeStar":3.5,
        "avergeStar":3.5,
        "keyCom":{
            "keyCom1":"good",
            "keyCom2":"bad",
            "keyCom3":"good",
            "keyCom4":"good",
            "keyCom5":"good",
            "keyCom6":"good",
            "keyCom7":"good",
            "keyCom8":"good",
            "keyCom9":"good",
            "keyCom10":"done"
        }
    }
    }
    # r=requests.post("115.159.91.188:3000/statue",data=result,timeout=1)
    header={"content-type": "application/json"}
    r=requests.post("http://115.159.91.188:3000/statue",json=result,headers=header)
    # r=requests.post("10.180.149.125:8082/main",data=result)
    # r=requests.post("10.180.149.125:8082/main",json=json.dumps(result))

    return r.content

def process_example():
    records=json.load(open("./data/example.json"))
    d={}
    d['queryID']=1
    d["data"]=records
    pr(d)
    result= process(d)
    pr(result)
    return


if __name__ == '__main__':
    # server_ip='10.214.25.140'
    process_example()
    server_ip='10.189.154.125'
    server_port=8082
    print("happliy start up!")
    app.run( host=server_ip,port=server_port )
