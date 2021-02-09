# -*- Encoding: UTF-8 -*-

"""Helper functions to construct the analysis pipeline."""


from analysis.task_spectral import task_null, task_crosscorr
from analysis.task_spectral import task_crosspower, task_crossphase, task_coherence
from analysis.task_spectral_cy import task_coherence_cy, task_crosspower_cy, task_crossphase_cy
try:
    from analysis.task_spectral_cu import task_coherence_cu, task_crosscorr_cu, task_crossphase_cu, task_crosspower_cu
except ModuleNotFoundError:
    None

from analysis.task_spectral_numba import task_spectral_GAP


def get_analysis_task(key, params, cfg_storage):
    """Returns an instance of a callable tasks for a given key.

    Args:
        key (string):
            Name of the analysis routine
        params (dictionary):
            Dictionary containing kwargs for the analysis kernel.
        cfg_storage (dictionary):
            Storage section of the Delta config

    Returns:
        obj (callable):
            Callable analysis kernel object

    Raises:
        NameError:
            If the key can not be matched to an available analysis object.
    """

    if key == "null":
        return task_null(params, cfg_storage)
    elif key == "crosscorr":
        return task_crosscorr(params, cfg_storage)
    elif key == "crosscorr_cu":
        # This does not work if there is cupy is not installed
        try:
            return task_crosscorr_cu(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
    elif key == "crossphase":
        return task_crossphase(params, cfg_storage)
    elif key == "crossphase_cu":
        try:
            return task_crossphase_cu(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
    elif key == "crossphase_cy":
        return task_crossphase_cy(params, cfg_storage)  
    elif key == "crosspower":
        return task_crosspower(params, cfg_storage)
    elif key == "crosspower_cy":
        return task_crosspower_cy(params, cfg_storage)
    elif key == "crosspower_cu":
        try:
            return task_crosspower_cu(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
    elif key == "coherence":
        return task_coherence(params, cfg_storage)
    elif key == "coherence_cy":
        return task_coherence_cy(params, cfg_storage)
    elif key == "coherence_cu":
        try:
            return task_coherence_cu(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
    elif key == "spectral_GAP":
        return task_spectral_GAP(params, cfg_storage)
    else:
        raise NameError(f"Requested invalid analysis routine: {key}")

# End of file analysis/helpers.py
