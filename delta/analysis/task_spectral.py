# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral import kernel_null, kernel_crosscorr
from analysis.kernels_spectral import kernel_coherence, kernel_crossphase, kernel_crosspower
import ray 


@ray.remote
class task_null(task_base):
    """Does nothing."""
    def __str__(self):
        return "task_null"

    def _get_kernel(self):
        return kernel_null

@ray.remote
class task_crosscorr(task_base):
    """Calculcates cross-correlation using numpy kernel."""
    def __str__(self):
        return "task_crosscorr"

    def _get_kernel(self):
        return kernel_crosscorr


@ray.remote
class task_coherence(task_base):
    """Calculcates coherence using numpy kernel."""
    def __str__(self):
        return "task_coherence"

    def _get_kernel(self):
        return kernel_coherence

    
@ray.remote
class task_crossphase(task_base):
    """Calculates crossphase using numpy kernel."""
    def __str__(self):
        return "task_crossphase"

    def _get_kernel(self):
        return kernel_crossphase
    
    
@ray.remote
class task_crosspower(task_base):
    """Calculates crosspower using numpy kernel."""
    def __str__(self):
        return "task_crosspower"
        
    def _get_kernel(self):
        return kernel_crosspower

    
# End of file task_spectral.py
