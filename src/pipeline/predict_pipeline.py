import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        cement: int,
        blast_furnace_slag: int, 
        fly_ash: int,           
        water: int,           
        superplasticizer: int, 
        coarse_aggregate: int,
        fine_aggregate : int,
        age: int):

        self.cement = cement

        self.blast_furnace_slag = blast_furnace_slag

        self.fly_ash = fly_ash

        self.water = water

        self.superplasticizer = superplasticizer

        self.coarse_aggregate = coarse_aggregate

        self.fine_aggregate  = fine_aggregate

        self.age = age 

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "cement": [self.cement],
                "blast_furnace_slag": [self.blast_furnace_slag],
                "fly_ash": [self.fly_ash],
                "water": [self.water],
                "superplasticizer": [self.superplasticizer],
                "coarse_aggregate": [self.coarse_aggregate],
                "fine_aggregate ": [self.fine_aggregate ],
                "age": [self.age],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)