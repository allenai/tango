from tango.integrations.flax.optim import LRScheduler, Optimizer


def test_all_optimizers_registered():
    assert "optax::adafactor" in Optimizer.list_available()


def test_all_lr_schedulers_registered():
    assert "optax::constant_schedule" in LRScheduler.list_available()
