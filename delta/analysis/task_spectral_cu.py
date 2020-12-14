# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral_cu import kernel_crosscorr_cu, kernel_coherence_cu
from analysis.kernels_spectral_gpu import kernel_crosspower_cu, kernel_crossphase_cu
    

class task_crosscorr_cu(task_base):
    """Calculates cross-correlation using CuPy kernel."""
    def __str__(self):
        return "task_crosscorr_cu"

    def _get_kernel(self):
        return kernel_crosscorr_cu


class task_coherence_cu(task_base):
    """Calculates coherence using CuPy kernel."""
    def __str__(self):
        return "task_coherence_cu"

    def _get_kernel(self):
        return kernel_coherence_cu


class task_crosspower_cu(task_base):
    """Calculates cross-power using CuPy kernel."""
    def __str__(self):
        return "task_crosspower_cu"

    def _get_kernel(self):
        return kernel_crosspower_cu


class task_crossphase_cu(task_base):
    """Calculates cross-phase using CuPy kernel."""
    def __str__(self):
        return "task_crossphase_cu"
        
    def _get_kernel(self):
        return kernel_crossphase_cu


# End of file task_spectral_cupy.py
