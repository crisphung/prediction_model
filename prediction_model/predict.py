import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline,load_dataset


classification_pipeline = load_pipeline()

def generate_predict(data_input):
    data = pd.DataFrame(data_input)

    pred = classification_pipeline.predict(data[config.FEATURES])

    output = np.where(pred == 1,'Y','N')

    result = {"prediction": output}
    return result

def generate_predict_data_name():
    
    test_data = load_dataset(config.TEST_FILE)

    pred = classification_pipeline.predict(test_data[config.FEATURES])

    output = np.where(pred == 1,'Y','N')
    print(output)

    #result = {"Predictions": output}
    return output

if __name__ == "__main__":
    generate_predict()

