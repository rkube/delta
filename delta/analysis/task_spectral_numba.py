# -*- Encoding: UTF-8 -*-

from analysis.task_base import task_base
from analysis.kernels_spectral_gpu import kernel_spectral_GAP


class task_spectral_GAP(task_base):
    """Calculates coherence, cross-power and cross-phase in a fused kernel."""
    def _get_kernel(self):
        return kernel_spectral_GAP