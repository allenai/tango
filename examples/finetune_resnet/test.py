from components import Model


def test_model_from_params():
    Model.from_params({
        "type": "resnet_ft", 
        "num_classes": 2, 
        "feature_extract": True, 
        "use_pretrained": True
    })
