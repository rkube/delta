# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral import kernel_null, kernel_crosscorr
from analysis.kernels_spectral import kernel_coherence, kernel_crossphase, kernel_crosspower


class task_null(task_base):
    """Does nothing."""
    def __str__(self):
        return "task_null"

    def _get_kernel(self):
        return kernel_null


class task_crosscorr(task_base):
    """Calculcates cross-correlation using numpy."""
    def __str__(self):
        return "task_crosscorr"

    def _get_kernel(self):
        return kernel_crosscorr


class task_coherence(task_base):
    """Calculcates coherence using numpy."""
    def __str__(self):
        return "task_coherence"

    def _get_kernel(self):
        return kernel_coherence


class task_crossphase(task_base):
    """Calculates crossphase using numpy."""
    def __str__(self):
        return "task_crossphase"

    def _get_kernel(self):
        return kernel_crossphase


class task_crosspower(task_base):
    """Calculates crosspower using numpy."""
    def __str__(self):
        return "task_crosspower"
        
    def _get_kernel(self):
        return kernel_crosspower

