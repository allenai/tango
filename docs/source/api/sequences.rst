Sequences
=========

This module contains some utilities to make sequences out of other sequences. All of these are lazy, so they
take minimal time and memory when you create them. These work particularly well when used together. For example,
you can concatenate two sequences (:class:`~tango.common.sequences.ConcatenatedSequence`), and then shuffle
them (:class:`~tango.common.sequences.ShuffledSequence`).

This module is not dependent on other Tango modules and can be used in isolation.

.. automodule:: tango.common.sequences
   :members:
