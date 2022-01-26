Deterministic Hashing
=========

In order to detect whether a :class:`~tango.step.Step` has to be re-run or not, Tango relies on some tools to compute
deterministic hashes from the inputs to the :class:`~tango.step.Step`.

The center-piece of this module is the :func:`~tango.common.det_hash.det_hash` function, which computes a deterministic hash of an
arbitrary Python object. The other things in this module influence how that works in various ways.

.. automodule:: tango.common.det_hash
   :members:
