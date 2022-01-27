from components import Model

def test_model_from_params():
    # Model.from_params({"type": "gpt2", "pretrained_model_name_or_path": "sshleifer/tiny-gpt2"})
    Model.from_params({"type": "resnet_ft", "pretrained_model_name_or_path": "sshleifer/tiny-gpt2"})
