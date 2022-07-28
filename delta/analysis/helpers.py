# -*- Encoding: UTF-8 -*-

"""Helper functions to construct the analysis pipeline."""

from analysis.task_spectral import task_null, task_crosscorr
from analysis.task_spectral import task_crosspower, task_crossphase, task_coherence
#from analysis.task_spectral_cy import task_coherence_cy, task_crosspower_cy, task_crossphase_cy
from analysis.task_spectral_cu import task_coherence_cu, task_crosscorr_cu, task_crossphase_cu, task_crosspower_cu
#from analysis.task_spectral_numba import task_spectral_GAP

import ray


@ray.remote
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
        return task_null.remote(params, cfg_storage)
    
        #------crosscorr------        
    elif key == "crosscorr":
        return task_crosscorr.remote(params, cfg_storage)
    elif key == "crosscorr_cu":
        try:
            return task_crosscorr_cu.remote(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy") 
            
        #------crossphase------    
    elif key == "crossphase":
        return task_crossphase.remote(params, cfg_storage)
    elif key == "crossphase_cy":
        raise NameError("Ray-version of DELTA is not suitable to run Cython analysis kernels - try the standard version of DELTA")
    elif key == "crossphase_cu":
        try:
            return task_crossphase_cu.remote(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")

        #------crosspower------
    elif key == "crosspower":
        return task_crosspower.remote(params, cfg_storage)
    elif key == "crosspower_cy":
        raise NameError("Ray-version of DELTA is not suitable to run Cython analysis kernels - try the standard version of DELTA")
    elif key == "crosspower_cu":
        try:
            return task_crosspower_cu.remote(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
        
        #------coherence------
    elif key == "coherence":
        return task_coherence.remote(params, cfg_storage)
    elif key == "coherence_cy":
        raise NameError("Ray-version of DELTA is not suitable to run Cython analysis kernels - try the standard version of DELTA")
    elif key == "coherence_cu":
        try:
            return task_coherence_cu.remote(params, cfg_storage)
        except NameError:
            raise NameError(f"Requested invalid analysis routine: {key}. Please install cupy")
        
        
         #------spectral_GAP------
    elif key == "spectral_GAP":
        raise NameError("Ray-version of DELTA is not suitable to run GPU numba kernels - try the standard version of DELTA")
       
    
         #------Other routines------
    else:
        raise NameError(f"Requested invalid analysis routine: {key}")

        
# End of file analysis/helpers.py