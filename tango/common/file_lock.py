import os
import warnings

from filelock import AcquireReturnProxy
from filelock import FileLock as _FileLock
from filelock import Timeout

from .aliases import PathOrStr


class FileLock(_FileLock):  # type: ignore[valid-type,misc]
    """
    This is just a subclass of the `FileLock` class from the `filelock` library, except that
    it adds an additional argument to the `__init__` method: `read_only_ok`.

    By default this flag is `False`, which an exception will be thrown when a lock
    can't be acquired due to lack of write permissions.
    But if this flag is set to `True`, a warning will be emitted instead of an error when
    the lock already exists but the lock can't be acquired because write access is blocked.
    """

    def __init__(self, lock_file: PathOrStr, timeout=-1, read_only_ok: bool = False) -> None:
        super().__init__(str(lock_file), timeout=timeout)
        self._read_only_ok = read_only_ok

    def acquire(self, timeout=None, poll_interval=0.05) -> AcquireReturnProxy:
        try:
            return super().acquire(timeout=timeout, poll_interval=poll_interval)
        except OSError as err:
            # OSError could be a lot of different things, but what we're looking
            # for in particular are permission errors, such as:
            #  - errno 1  - EPERM  - "Operation not permitted"
            #  - errno 13 - EACCES - "Permission denied"
            #  - errno 30 - EROFS  - "Read-only file system"
            if err.errno not in (1, 13, 30):
                raise

            if os.path.isfile(self._lock_file) and self._read_only_ok:
                warnings.warn(
                    f"Lacking permissions required to obtain lock '{self._lock_file}'. "
                    "Race conditions are possible if other processes are writing to the same resource.",
                    UserWarning,
                )
                return None  # type: ignore[return-value]
            else:
                raise

    def acquire_with_updates(self, desc: str = "acquiring lock"):
        """
        Same as :meth:`acquire()`, except that when the lock cannot be immediately acquired,
        it will keep trying and print status updates as it goes.
        """
        try:
            return self.acquire(timeout=0.1)
        except Timeout:
            pass

        from .tqdm import Tqdm

        progress = Tqdm.tqdm(desc=desc, bar_format="{desc} [{elapsed}]")
        while True:
            progress.update()
            try:
                return self.acquire(timeout=1)
            except Timeout:
                continue
