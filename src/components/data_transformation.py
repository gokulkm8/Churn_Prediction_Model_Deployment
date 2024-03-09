import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def feature_engg(self,data_file_path):

        df=pd.read_csv(data_file_path)
        try:
            df.drop(['Unnamed: 0'],axis=1,inplace=True)
            new_column_names = {'MMM-YY': 'MMM_YY', 'Joining Designation': 'Joining_Designation',
                            'Total Business Value':'Total_Business_Value', 'Quarterly Rating':'Quarterly_Rating'
                            }
            df.rename(columns=new_column_names, inplace=True)

        except:
            pass


        if len(df['Dateofjoining'][0])>8:
            try:
                df['Dateofjoining']=pd.to_datetime(df['Dateofjoining'],format='%d/%m/%Y').dt.strftime('%d/%m/%y')
            except:
                df['Dateofjoining']=pd.to_datetime(df['Dateofjoining'],format='%Y-%m-%d').dt.strftime('%d/%m/%y')
        

        df['Dateofjoining']=pd.to_datetime(df['Dateofjoining'],format='%d/%m/%y')

        df['Joining_Designation']=df['Joining_Designation'].astype('object')
        df['Grade']=df['Grade'].astype('object')

        df_temp = df.groupby('Driver_ID')['Quarterly_Rating'].agg(['first','last']).reset_index()

        df_temp['imp']=np.where(df_temp['last']>df_temp['first'],1,0)

        ids = df_temp[df_temp['imp']==1]['Driver_ID']

        df['quarterly_rating_imp']=0

        df.loc[df['Driver_ID'].isin(ids),'quarterly_rating_imp']=1

        df_temp = df.groupby('Driver_ID')['Income'].agg(['first','last']).reset_index()

        df_temp['imp']=np.where(df_temp['last']>df_temp['first'],1,0)

        ids = df_temp[df_temp['imp']==1]['Driver_ID']

        df['income_imp']=0

        df.loc[df['Driver_ID'].isin(ids),'income_imp']=1

        dictionary = {'MMM_YY':'max','Age':'max','Gender':'last','City':'last','Education_Level':'last',
                'Income':'last','Dateofjoining':'last','LastWorkingDate':'last','Joining_Designation':'last',
                'Grade':'last','Total_Business_Value':'sum','Quarterly_Rating':'last','quarterly_rating_imp':'last',
                'income_imp':'last'}

        df_grouped = df.groupby('Driver_ID').aggregate(dictionary).reset_index()

        df_grouped['monthofjoining'] = df_grouped['Dateofjoining'].dt.month

        df_grouped['Target'] = np.where(df_grouped['LastWorkingDate'].isnull(),0,1)
        
        df_grouped.drop(['Driver_ID','MMM_YY','Gender','Education_Level','Dateofjoining',
                        'LastWorkingDate'],axis=1,inplace=True)

        return df_grouped
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:

            numerical_columns = ["Income", "Age","Joining_Designation","Grade","Total_Business_Value",
                                "Quarterly_Rating","quarterly_rating_imp","income_imp","monthofjoining"]
            categorical_columns = ["City"]

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("target encoding",TargetEncoder()),
                ("scaler",StandardScaler())
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Target"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(X=input_feature_train_df,y=target_feature_train_df)

            smt=SMOTE()
            input_feature_train_arr,target_feature_train_df=smt.fit_resample(X=input_feature_train_arr,y=target_feature_train_df)

            input_feature_test_arr=preprocessing_obj.transform(X=input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)