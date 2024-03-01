from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            cement=float(request.form.get('cement')),
            blast_furnace_slag=float(request.form.get('blast_furnace_slag')),
            fly_ash =float(request.form.get('fly_ash')),
            water=float(request.form.get('water')),
            superplasticizer=float(request.form.get('superplasticizer')),
            coarse_aggregate=float(request.form.get('coarse_aggregate')),
            fine_aggregate =float(request.form.get('fine_aggregate')),
            age=float(request.form.get('age'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        