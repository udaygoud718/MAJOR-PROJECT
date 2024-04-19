#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
from Feature import FeatureExtraction

file = open("Boost.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        #print(x)
        y_pred =gbc.predict(x)[0]
        #print(y_pred)
        return render_template('index.html',xx =y_pred,url=url )
    elif request.method=="GET":
       return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)