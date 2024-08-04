
import sys
import os
from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp
from sklearn.pipeline import Pipeline
from prediction_model.config import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

classification_pipeline = Pipeline([
    ('Mean imputation',pp.MeanImputer(variables=config.NUM_FEATURES)),
    ('Mode imputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
    ('Domain procesing',pp.DomainProcessing(var_to_modify=config.FEATURE_TO_MODIFY,var_to_add=config.FEATURE_TO_ADD)),
    ('Drop Features',pp.DropColumns(drop_cols=config.DROP_FEATURES)),
    ('Category encoder',pp.CategoryEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('Log transform',pp.LogTransform(variables=config.LOG_FEATURES)),
    ('Minmax scaler',MinMaxScaler()),
    ('Logistic Classifier',LogisticRegression(random_state=777))
])


