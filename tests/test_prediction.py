import pytest 
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset,load_pipeline
from prediction_model.predict import generate_predict


# output from predict script not null

# output from predict script is str data type

# output i Y for an example data

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)

    single_row = test_dataset[:1]

    result = generate_predict(single_row)

    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'