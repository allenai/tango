from tango.integrations.torch.optim import LRScheduler, Optimizer


def test_all_optimizers_registered():
    assert "torch::Adagrad" in Optimizer.list_available()


def test_all_lr_schedulers_registered():
    assert "torch::ExponentialLR" in LRScheduler.list_available()
