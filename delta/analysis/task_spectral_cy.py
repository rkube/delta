# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral_cy import kernel_coherence_64_cy
from analysis.kernels_spectral_cy import kernel_crosspower_64_cy, kernel_crossphase_64_cy


class task_coherence_cy(task_base):
    """Calculates coherence using Cython kernel."""
    def __str__(self):
        return "task_coherence_cy"

    def _get_kernel(self):
        return kernel_coherence_64_cy


class task_crosspower_cy(task_base):
    """Calculates crosspower using Cython kernel."""
    def __str__(self):
        return "task_crosspower_cy"

    def _get_kernel(self):
        return kernel_crosspower_64_cy


class task_crossphase_cy(task_base):
    """Calculates cross-phase using Cython kernel."""
    def __str__(self):
        return "task_crossphase_cy"
        
    def _get_kernel(self):
        return kernel_crossphase_64_cy
