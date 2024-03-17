import os
import sys
from scipy.stats import ttest_ind,chi2_contingency
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
    
class FeatureEngineering:

    def __init__(self):
        pass

    def feature_extraction(self,df):

        try:

            logging.info("Feature extraction initiated")

            #numerical_columns = ["Income", "Age","Joining_Designation","Grade","Total_Business_Value",
                                #"Quarterly_Rating","quarterly_rating_imp","income_imp","monthofjoining"]
            #categorical_columns = ["City",'Gender','Education_Level']

            numerical_columns=[]

            categorical_columns=[]

            for i in df.drop('Target',axis=1).columns:
                if df[i].nunique()>30:
                    numerical_columns.append(i)
                else:
                    categorical_columns.append(i)
            print(numerical_columns)
            print(categorical_columns)
            # Hypothesis testing ( t-test ) to check statistical significance of numerical columns
            # on predicting the target feature

            for i in numerical_columns:
                feat_0=df[df['Target']==0][i]
                feat_1=df[df['Target']==1][i]

                t_stat,p_val = ttest_ind(feat_0,feat_1)

                if p_val > 0.05:
                    df.drop(i,axis=1,inplace=True)

            # Hypothesis testing ( chi-square ) to check statistical significance of categorical columns
            # on predicting the target feature

            for i in categorical_columns:

                cross_tab=pd.crosstab(index=df[i],columns=df['Target'])

                chi_stat,p_val,dof,expected_val=chi2_contingency(cross_tab)

                if p_val>0.05:
                    df.drop(i,axis=1,inplace=True)

            logging.info("Feature extraction completed")
            
            return df

        except Exception as e:
            raise CustomException(e,sys)

    def feature_engg(self,data_file_path):

        try:

            logging.info("Feature engineering initiated")

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
            
            df_grouped.drop(['Driver_ID','MMM_YY','Dateofjoining','LastWorkingDate'],axis=1,inplace=True)
            
            df_grouped = self.feature_extraction(df_grouped)

            logging.info("Feature engineering completed")


            return df_grouped

        except Exception as e:
            raise CustomException(e,sys)
        