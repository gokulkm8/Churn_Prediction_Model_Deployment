from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_data():

    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            MMM_YY =  request.form.get('MMM_YY'),
            Driver_ID = request.form.get('Driver_ID') ,
            Age = request.form.get('Age') , 
            Gender = request.form.get('Gender')  ,
            City = request.form.get('City')  ,
            Education_Level = request.form.get('Education_Level') , 
            Income = request.form.get('Income') ,
            Dateofjoining = request.form.get('Dateofjoining') ,
            LastWorkingDate = request.form.get('LastWorkingDate') ,
            Joining_Designation = request.form.get('Joining_Designation') ,
            Grade = request.form.get('Grade'),
            Total_Business_Value = request.form.get('Total_Business_Value'),
            Quarterly_Rating = request.form.get('Quarterly_Rating'),
        )
        pred_df_path = data.get_data_as_data_frame()
        results = PredictPipeline().predict(pred_df_path) 
        data_with_pred = pd.read_csv(pred_df_path)
        data_with_pred['Target'] = results[0]
        data_collection = pd.read_csv(r'new_data_artifacts\new_data_collection.csv')
        pd.concat([data_collection,data_with_pred]).to_csv(r'new_data_artifacts\new_data_collection.csv',index=False)

        if results[0]==0:
            return_result = "The Driver will stay in the company"
        else:
            return_result = "The Driver will leave the company"

        return render_template('home.html', results=return_result)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)  