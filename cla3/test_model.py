from model import train_and_evaluate

def test_model_can_make_predictions():
    model, encoder, features, accuracy = train_and_evaluate()
    assert hasattr(model, "predict")
    assert len(features) == 4

def test_model_accuracy_is_reasonable():
    model, encoder, features, accuracy = train_and_evaluate()
    assert accuracy >= 0.80
