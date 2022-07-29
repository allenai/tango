Global settings
---------------

Some command-line options can set globally in a ``tango.yml`` or ``tango.yaml`` settings file.
Tango will check the current directory and ``~/.config/``, in that order.

The full spec of this config is defined by the :class:`~tango.settings.TangoGlobalSettings` class.

.. autoclass:: tango.settings.TangoGlobalSettings
   :members:
   :exclude-members: path,find_or_default
   :member-order: bysource
