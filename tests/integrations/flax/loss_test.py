from tango.integrations.flax.loss import LossFunction


def test_all_losses_registered():
    assert "optax::softmax_cross_entropy" in LossFunction.list_available()
