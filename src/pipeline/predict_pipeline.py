import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from src.components.data_feature_engg import FeatureEngineering
from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
    new_data_path = os.path.join("new_data_artifacts","new_data.csv")

class PredictPipeline:
    def __init__(self):
        self.data_path = PredictPipelineConfig()

    def predict(self,new_data_file_path):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            feature_engg_data=FeatureEngineering().feature_engg(new_data_file_path)
            data_scaled=preprocessor.transform(feature_engg_data)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(   self,
                    MMM_YY, 
                    Driver_ID, 
                    Age, 
                    Gender, 
                    City,
                    Education_Level,
                    Income,
                    Dateofjoining,
                    LastWorkingDate,
                    Joining_Designation, 
                    Grade,
                    Total_Business_Value,
                    Quarterly_Rating

                ):

        self.MMM_YY =  MMM_YY
        self.Driver_ID = Driver_ID 
        self.Age = Age 
        self.Gender = Gender 
        self.City = City
        self.Education_Level = Education_Level
        self.Income = Income
        self.Dateofjoining = Dateofjoining
        self.LastWorkingDate = LastWorkingDate
        self.Joining_Designation = Joining_Designation 
        self.Grade = Grade
        self.Total_Business_Value = Total_Business_Value
        self.Quarterly_Rating = Quarterly_Rating

    def get_data_as_csv_path(self):
        try:
            custom_data_input_dict = {
                "MMM_YY" : [self.MMM_YY],
                "Driver_ID": [self.Driver_ID],
                "Age": [self.Age],
                "Gender": [self.Gender],
                "City": [self.City],
                "Education_Level": [self.Education_Level],
                "Income": [self.Income],
                "Dateofjoining": [self.Dateofjoining],
                "LastWorkingDate": [self.LastWorkingDate],
                "Joining_Designation": [self.Joining_Designation],
                "Grade": [self.Grade],
                "Total_Business_Value": [self.Total_Business_Value],
                "Quarterly_Rating": [self.Quarterly_Rating]

            }
            df_custom_data=pd.DataFrame(custom_data_input_dict)
            predict_pipeline_config = PredictPipeline()
            os.makedirs(os.path.dirname(predict_pipeline_config.data_path.new_data_path),exist_ok=True)
            df_custom_data.to_csv(predict_pipeline_config.data_path.new_data_path,index=False,header=True)
        
            return predict_pipeline_config.data_path.new_data_path

        except Exception as e:
            raise CustomException(e, sys)