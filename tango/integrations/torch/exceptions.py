from tango.common.exceptions import TangoError


class StopEarly(TangoError):
    """
    Callbacks can raise this exception to stop training early without crashing.

    .. important::
        During distributed training all workers must raise this exception at the same point
        in the training loop, otherwise there will be a deadlock.
    """
