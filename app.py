import json
import pickle

from markupsafe import escape as jinja_escape
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('model.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    input_data = np.array(list(data.values())).reshape(1, -1)
    # print(input_data)
    # print(np.array(list(input_data.values())).reshape(1,-1))
    # new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(input_data)
    output_int = int(output[0])
    return jsonify(output_int)

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
   
     