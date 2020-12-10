# -*- Encoding: UTF-8 -*-

"""Implements analysis task list."""

import logging
from analysis.helpers import get_analysis_task


class tasklist():
    """Defines an analysis task list.

    This class defines an task list that is executed in parallel on an executor.
    """

    def __init__(self, executor, cfg):
        """Configures the analysis tasklist from a dictionary.

        For each key-value pair in "cfg_analysis", an analysis task is instantiated
        and appended to a list. On a call to execute, all tasks are launched with
        the current data.

        Args:
            executor (PEP-3148 style executor):
                Executor on which all analysis tasks are launched
            cfg:
                Delta configuration

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")
        self.cfg = cfg
        self.executor = executor
        self.tasklist = []
        for key, anl_params in cfg["analysis"].items():
            try:
                self.tasklist.append(get_analysis_task(key, anl_params, cfg["storage"]))
            except NameError as e:
                self.logger.error(f"Could not find a suitable analysis task: {e}")
                continue
            self.logger.info(f"Added {key} to analysis task list")

    def execute(self, timechunk):
        """Execute all analysis tasks.

        Args:
            timechunk (timechunk):
                A time-chunk of 2D image data.

        Returns:
            None
        """
        self.logger.info(f"Submitting timechunk {timechunk.tb.chunk_idx} to analysis tasklist")
        for task in self.tasklist:
            task.execute(timechunk, self.executor)
        
# End of file task_list.py
#             