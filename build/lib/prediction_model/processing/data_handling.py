import os
import pandas as pd
import joblib

from prediction_model.config import config

def load_dataset(file_name):

    file_path = os.path.join(config.DATA_PATH,file_name)
    _data = pd.read_csv(file_path)
    return _data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

def load_pipeline():
    load_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    pipe = joblib.load(load_path)
    print(f"Model has been loaded from the name {config.MODEL_NAME}")
    return pipe





