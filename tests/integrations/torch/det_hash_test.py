import numpy
import torch

from tango.common import det_hash


def test_numpy_det_hash():
    a1 = numpy.array([[1, 2], [3, 4]], order="C")
    a2 = numpy.array([[1, 2], [3, 4]], order="K")
    assert det_hash(a1) == det_hash(a2)


def test_torch_det_hash():
    a1 = numpy.array([[1, 2], [3, 4]], order="C")
    a2 = numpy.array([[1, 2], [3, 4]], order="K")
    a1 = torch.tensor(a1)
    a2 = torch.tensor(a2)
    assert det_hash(a1) == det_hash(a2)
