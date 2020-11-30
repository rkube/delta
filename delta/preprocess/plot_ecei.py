# -*- Encoding: UTF-8 -*-

"""Plotting methods for ECEI data."""

import logging

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# from data_models.kstar_ecei import ecei_chunk
from data_models.kstar_ecei import get_geometry


class radial_interpolator():
    """Defines radial interpolation, based on values of channels close-by."""

    def __init__(self, rpos_arr, zpos_arr, mask):
        """Initializes radial interpolator.

        Args:
            rpos_arr (ndarray, float)
                shape=(24, 8): radial position of all ECEI views, in m
            zpos_arr (ndarray, float):
                shape=(24, 8): vertical position of all ECEI views, in m
            mask (ndarray, bool):
                shape=(24, 8): True marks bad pixel

        Returns:
            None
        """
        self.rpos_arr = rpos_arr
        self.zpos_arr = zpos_arr

    def __call__(self, frame, cutoff=0.03):
        """Applies radial interpolation on a frame.

        Args:
            frame (ndarray,float):
                Frame values
            mask (ndarray,bool):
                True marks channels that have good data. False marks channels
                with bad data.
            cutoff (float):
                scale-length for mask kernel around bad channels

        Returns:
            ndarray(float):
        """
        # frame_up will be the output array
        frame_ip = np.copy(frame)
        # Channels that have bad data are initialized to zero
        frame_ip[mask] = 0.0
        # So that we don't use interpolated data from one channel when we
        # interpolate for another channel we create another instance of
        # the frame with zeros in the bad channels. This array will not get
        # updated.
        frame_zeros = np.copy(frame)
        frame_zeros[mask] = 0.0

        for bad_idx in np.argwhere(mask):
            r_bad = self.rpos_arr[bad_idx[0], bad_idx[1]]
            z_bad = self.zpos_arr[bad_idx[0], bad_idx[1]]
            dist = np.linalg.norm(np.stack((self.rpos_arr - r_bad,
                                           self.zpos_arr - z_bad)), axis=0)
            dfct = np.exp(-2. * (dist / cutoff) ** 4.) * frame_zeros

            frame_ip[bad_idx[0], bad_idx[1]] = (frame_zeros * dfct).sum() / dfct.sum()

        return frame_ip


class plot_ecei_timeslice():
    """Plot a time slice of an ecei view."""
    def __init__(self, chunk, cmap=plt.cm.RdYlBu):
        """Initializes the plotting class.

        Args:
            time_chunk (time-chunk):
                time-chunk of ECEI data.
            interpolator (???):
                Interpolator type

        Returns:
            None
        """
        self.logger = logging.getLogger("simple")

        self.clevs = np.linspace(-0.15, 0.15, 64)
        self.rpos_arr, self.zpos_arr, _ = get_geometry(chunk.params)
        self.bad_channels = chunk.bad_channels
        self.cmap = cmap
        self.clevs = None
        self.interpolator = None

        # self._set_contour_levels(chunk, 64)

        mpl.use("AGG")


    def create_plot(self, chunk, tidx):
        """Creates contour plots in the time-chunk of ECEI data.

        Args:
            chunk (???):
                Time-chunk ECEI data.
            tb (timebase_streaming):
                Timebase for the ECEI data

        Returns:
            fig (mpl.Figure):
                Matplotlib figure
        """
        frame_vals = chunk.data[:, tidx]

        if (~chunk.bad_channels).sum() == 0:
            all_max = chunk.data[:, tidx].max()
            all_min = chunk.data[:, tidx].min()
        else:
            all_max = chunk.data[~chunk.bad_channels, tidx].max()
            all_min = chunk.data[~chunk.bad_channels, tidx].min()
        #self.clevs = np.linspace(all_min, all_max, 64)
        self.clevs = np.linspace(-0.3, 0.3, 64)

        if self.interpolator is not None:
            self.logger.info("Interpolating frames for plotting")
            frame_vals = self.interpolator(frame_vals, mask=chunk.bad_channels)

        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.46, 0.75])
        ax_cb = fig.add_axes([0.7, 0.2, 0.05, 0.75])

        t0, _ = chunk.tb.get_trange()
        time = t0 + chunk.tb.dt * tidx
        title_str = f"t = {time:8.6f}s"

        mappable = None
        if self.rpos_arr is not None and self.zpos_arr is not None:
            self.logger.info(f"Plotting data")  #: {frame_vals.reshape(24, 8)}")
            self.logger.info(f"Using contour levels {self.clevs[0]}, {self.clevs[-1]}")
            # TODO: Fix hard-coded dimensions
            mappable = ax.contourf(self.rpos_arr.reshape(24, 8),
                                   self.zpos_arr.reshape(24, 8),
                                   frame_vals.reshape(24, 8), levels=self.clevs,
                                   cmap=self.cmap)

            ax.set_xlabel("R / m")
            ax.set_ylabel("Z / m")
            ax.set_title(title_str)
        # else:
        #   mappable = ax.contourf(self.rpos_arr, self.zpos_arr, frame_vals)#, levels=self.clevs)
        fig.colorbar(mappable, cax=ax_cb)

        return fig

# End of file plot_ecei.py
