# Encoding: UTF-8 -*-

"""Storage backend.

Defines a basic interface to the backend-storage classes and helper routines
"""

from storage.backend_mongodb import backend_mongodb
from storage.backend_numpy import backend_numpy
from storage.backend_null import backend_null


def get_storage_object(cfg_storage):
    """Returns the storage class matching name.

    Args:
        cfg_storage (dict):
            Delta configuration, storage section

    Returns:
        backend (storage):
            Storage object

    Raises:
        NameError:
            In case where no storage backend can be associated with the name.
    """
    if cfg_storage["backend"] == "numpy":
        return backend_numpy
    elif cfg_storage["backend"] == "mongo":
        return backend_mongodb
    elif cfg_storage["backend"] == "null":
        return backend_null
    else:
        raise NameError("Unknown storage backend requested: " + cfg_storage["backend"])


# End of file backend.py
