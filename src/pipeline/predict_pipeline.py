import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


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
        brand: str,
        storage: float,
        ram: float,
        screen_size: float,
        camera: float,
        battery_capacity: float,
        camera_count: float,
        ):

        self.brand = brand

        self.storage = storage

        self.ram = ram

        self.screen_size = screen_size

        self.camera = camera

        self.battery_capacity = battery_capacity



        self.camera_count = camera_count


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "storage": [self.storage],
                "ram": [self.ram],
                "screen_size": [self.screen_size],
                "camera": [self.camera],
                "battery_capacity": [self.battery_capacity],
                "camera_count": [self.camera_count]
                }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

