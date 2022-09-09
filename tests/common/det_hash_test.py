from tango.common.det_hash import DetHashWithVersion, det_hash


def test_normal_det_hash():
    class C:
        VERSION = 1

        def __init__(self, x: int):
            self.x = x

    c1_1 = C(10)
    c2_1 = C(10)
    c3_1 = C(20)
    assert det_hash(c1_1) == det_hash(c2_1)
    assert det_hash(c3_1) != det_hash(c2_1)

    class C:
        VERSION = 2

        def __init__(self, x: int):
            self.x = x

    c1_2 = C(10)
    c2_2 = C(10)
    c3_2 = C(20)
    assert det_hash(c1_2) == det_hash(c2_2)
    assert det_hash(c3_2) != det_hash(c2_2)
    assert det_hash(c1_2) == det_hash(c1_1)  # because the version isn't taken into account
    assert det_hash(c3_2) == det_hash(c3_1)  # because the version isn't taken into account


def test_versioned_det_hash():
    class C(DetHashWithVersion):
        VERSION = 1

        def __init__(self, x: int):
            self.x = x

    c1_1 = C(10)
    c2_1 = C(10)
    c3_1 = C(20)
    assert det_hash(c1_1) == det_hash(c2_1)
    assert det_hash(c3_1) != det_hash(c2_1)

    class C(DetHashWithVersion):
        VERSION = 2

        def __init__(self, x: int):
            self.x = x

    c1_2 = C(10)
    c2_2 = C(10)
    c3_2 = C(20)
    assert det_hash(c1_2) == det_hash(c2_2)
    assert det_hash(c3_2) != det_hash(c2_2)
    assert det_hash(c1_2) != det_hash(c1_1)  # because the version is taken into account
    assert det_hash(c3_2) != det_hash(c3_1)  # because the version is taken into account


def test_numpy_det_hash():
    import numpy

    a1 = numpy.array([[1, 2], [3, 4]], order="C")
    a2 = numpy.array([[1, 2], [3, 4]], order="K")
    assert det_hash(a1) == det_hash(a2)


def test_torch_det_hash():
    import numpy
    import torch

    a1 = numpy.array([[1, 2], [3, 4]], order="C")
    a2 = numpy.array([[1, 2], [3, 4]], order="K")
    a1 = torch.tensor(a1)
    a2 = torch.tensor(a2)
    assert det_hash(a1) == det_hash(a2)
