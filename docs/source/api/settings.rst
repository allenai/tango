Global settings
---------------

Some command-line options can set globally in a ``tango.yml`` or ``tango.yaml`` settings file.
Tango will check the current directory and ``~/.config/``, in that order.

The full spec of this config is defined by :class:`TangoGlobalSettings`.

.. autoclass:: tango.settings.TangoGlobalSettings
   :members:
   :exclude-members: path,find_or_default
