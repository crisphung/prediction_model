import pathlib
import os
import prediction_model
import sys

from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


DATA_DIR = "datasets"
TRAINED_MODELS_DIR = "trained_models"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

MODEL_NAME = "classification.pkl"

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATA_PATH = os.path.join(PACKAGE_ROOT,DATA_DIR)


SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, TRAINED_MODELS_DIR)

TARGET = 'Loan_Status'

FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # taking log of numerical columns



