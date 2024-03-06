import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    raw_data_path = os.path.join('artifacts',"feature_engg_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def feature_engg(self):

        df=pd.read_csv('data\data.csv')
        df.drop(['Unnamed: 0'],axis=1,inplace=True)
        df['MMM-YY']=pd.to_datetime(df['MMM-YY'])
        df['Dateofjoining']=pd.to_datetime(df['Dateofjoining'])
        df['LastWorkingDate']=pd.to_datetime(df['LastWorkingDate'])
        df['Joining Designation']=df['Joining Designation'].astype('object')
        df['Grade']=df['Grade'].astype('object')

        df_temp = df.groupby('Driver_ID')['Quarterly Rating'].agg(['first','last']).reset_index()

        df_temp['imp']=np.where(df_temp['last']>df_temp['first'],1,0)

        ids = df_temp[df_temp['imp']==1]['Driver_ID']

        df['quarterly_rating_imp']=0

        df.loc[df['Driver_ID'].isin(ids),'quarterly_rating_imp']=1

        df_temp = df.groupby('Driver_ID')['Income'].agg(['first','last']).reset_index()

        df_temp['imp']=np.where(df_temp['last']>df_temp['first'],1,0)

        ids = df_temp[df_temp['imp']==1]['Driver_ID']

        df['income_imp']=0

        df.loc[df['Driver_ID'].isin(ids),'income_imp']=1

        dictionary = {'MMM-YY':'max','Age':'max','Gender':'last','City':'last','Education_Level':'last',
             'Income':'last','Dateofjoining':'last','LastWorkingDate':'last','Joining Designation':'last',
             'Grade':'last','Total Business Value':'sum','Quarterly Rating':'last','quarterly_rating_imp':'last',
             'income_imp':'last'}

        df_grouped = df.groupby('Driver_ID').aggregate(dictionary).reset_index()

        df_grouped['monthofjoining'] = df_grouped['Dateofjoining'].dt.month

        df_grouped['Target'] = np.where(df_grouped['LastWorkingDate'].isnull(),0,1)
        
        df_grouped.drop(['Driver_ID','MMM-YY','Gender','Education_Level','Dateofjoining',
                        'LastWorkingDate'],axis=1,inplace=True)

        return df_grouped


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=self.feature_engg()

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))