import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp
from prediction_model.pipeline import classification_pipeline
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,load_pipeline,save_pipeline



def perform_training():

    train_data = load_dataset(config.TRAIN_FILE)

    X_train = train_data[config.FEATURES]
    y_train = train_data[config.TARGET].map({'N':0,'Y':1})

    classification_pipeline.fit(X_train,y_train)

    save_pipeline(classification_pipeline)

if __name__=="__main__":
    perform_training()


