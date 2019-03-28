from flask import Flask, render_template, request, json, redirect
from werkzeug.utils import secure_filename
import pandas as pd
from test import MLEngine
import os.path
import json2table

import csv
app = Flask(__name__)
# t = Test()
# td = TestD()
# al = AutoLearn()
engine = MLEngine()
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def showjson():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "static/json", "train_1.json")
    # data = json.load(open(json_url))
    infoFromJson = json.load(open(json_url))
    build_direction = "LEFT_TO_RIGHT"
    table_attributes = {"style": "width:100%"}
    
    table_html = json2table.convert(infoFromJson, 
                                    build_direction=build_direction, 
                                    table_attributes=table_attributes)    
    return table_html
    
    # return render_template('showjson.jade', data=data)

# Define the basic route and its corresponding request handler
@app.route("/")
def main():
    #return "Welcome !"
    # print(showjson())
    table_html = showjson()
    return render_template('index.html', table_html = table_html)

@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    table_html = showjson()
    msg = "file uploaded successfully"
    print("request--", request.method)
    # if request.method == "POST":
    f = request.files["txnFile"]
    f.save(secure_filename(f.filename))
    print("file name->", f.filename)
    # else:
        # print("method not post")

    return render_template("uploaded.html", table_html = table_html, msg = msg)

@app.route("/checkFraud", methods=["GET", "POST"])
def checkFraud():
    # result = request.form
    df_test = {'step':'157', 'customer':'C790593937', 'age':'3', 'gender':'F', 'zipcodeOri':'28007', 'merchant':'M1823072687', 'zipMerchant':'28007', 'category':'es_transportation', 'amount':'16.98'}
    prediction = engine.test_forest(df_test)
    # return render_template("fraudResult.html", prediction = prediction)
    return render_template("fraudResult.html", prediction = prediction)

@app.route("/dlcheckFraud", methods=["GET","POST"])
def dlcheckFraud():
    # result = request.form
    # if request.method =='POST':
    # file = request.files['txnFile']
    # if file:
    #     filename = secure_filename(file.filename)
    #     print(filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  
    # else:
    #     print("no file found")  
   
    # df_test2 = {'step':'157', 'customer':'C790593937', 'age':'3', 'gender':'F', 'zipcodeOri':'28007', 'merchant':'M1823072687', 'zipMerchant':'28007', 'category':'es_transportation', 'amount':'16.98'}
    # prediction = td.dl_predict(pd.DataFrame(df_test, index=[0]))
    df_test2 = {'step':'0', 'customer':'C583110838', 'age':'3', 'gender':'M', 'zipcodeOri':'28807', 'merchant':'M480139044', 'zipMerchant':'358007', 'category':'es_transaction', 'amount':'16.26'}
    prediction = engine.predict(df_test2)
    # score = engine.getScore(df_test2)
    return render_template("fraudResult.html", prediction = prediction)
    
@app.route('/fraudResult')
def fraudResult():
    return render_template('fraudResult.html')


# Check if the executed file is the main program and run the app:
if __name__ == "__main__":
    # app.run(threaded=True)
    app.run(debug = True)

