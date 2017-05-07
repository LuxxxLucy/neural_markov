#encoding=utf-8

from flask import Flask, request
import os
from pprint import pprint as pr
import requests

from preprocessing import *

app = Flask(__name__)


@app.route("/",method=['POST'])
def index():
    # return app.send_static_file('base.html')
    for it in request.files:
        f=request.files[it]
        f.save('./'+"temp_working.json")
        records=json.load(open("./temp_working.json"))

        result= process(1112,records)

        r=requests.post("115.159.91.188/statue",data=result)

    return "OKay"


@app.route("/test",method=['POST','GET'])
def index():
    send_test()
    return "OKay"
def send_test():

    result={
    "queryID": 1000
    "statue": 'ok',
    data:{
        "commentsTotalNum": 10000,
        "validComNum": 7500,
        "invalidComNum": 2500,
        "validPercent":75,
        "invalidPercent":25,
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
    r=requests.post("115.159.91.188:3000/statue",data=result)

    return

if __name__ == '__main__':
    server_ip='10.214.25.140'
    server_port=8082
    print("happliy start up!")
    app.run( host=server_ip,port=server_port )
