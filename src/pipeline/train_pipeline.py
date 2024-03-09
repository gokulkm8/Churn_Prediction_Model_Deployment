import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class TrainPipeline:
    def __init__(self):
        pass

    def predict(self,datas):
        try:
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)