#-*- Encoding: UTF-8 -*-

import numpy as np

import matplotlib as mpl 
import matplotlib.pyplot as plt

from ecei_channel import ecei_view

class ecei_interpolator():
    def __init__(self):
        pass

    def __call__(self, frame):
        return frame


class radial_interpolator(ecei_interpolator):
    """
    Defines radial interpolation, based on values of channels close-by.
    """
    def __init__(self, rpos_arr, zpos_arr):
        """
        rpos_arr: ndarray, float: shape=(24, 8): radial position of all ECEI views, in m
        zpos_arr: ndarray, float: shape=(24, 8): vertical position of all ECEI views, in m
        """
        self.rpos_arr = rpos_arr
        self.zpos_arr = zpos_arr



    def __call__(self, frame, mask=None, cutoff=0.03):
        """Applies radial interpolation on the frame.
        frame: ndarray(float): Frame values
        mask: ndarray(bool): True marks channels that have good data. False marks channels
                             with bad data.
        cutoff: float, scale-length for mask kernel around bad channels
        
        Returns:
        --------
        ndarray(float)
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
            print(bad_idx)
            
            r_bad = self.rpos_arr[bad_idx[0], bad_idx[1]]
            z_bad = self.zpos_arr[bad_idx[0], bad_idx[1]]
            dist = np.linalg.norm(np.stack((self.rpos_arr - r_bad,
                                           self.zpos_arr - z_bad)), axis=0)
            dfct = np.exp(-2. * (dist / cutoff) ** 4.) * frame_zeros

            frame_ip[bad_idx[0], bad_idx[1]] = (frame_zeros * dfct).sum() / dfct.sum()

        return frame_ip


class plot_ecei_timeslice():
    """Plot a time slice of an ecei view"""
    def __init__(self, ecei_view, interpolator=None, rpos_arr=None, zpos_arr=None, cmap=plt.cm.RdYlBu):
        """
        Parameters:
        ===========
        interpolator:  an interpolator type
        """

        if interpolator is not None:
            self.interpolator = interpolator(rpos_arr, zpos_arr)

        self.clevs = np.linspace(-0.15, 0.15, 64)
        self.zpos_arr = zpos_arr
        self.rpos_arr = rpos_arr
        self.cmap = cmap

        self.clevs = None


    def set_contour_levels(self, ecei_view, nlevs=64):
        """Automatically determine the contour levels used for plotting."""

        # Slow code: 
        # # If we don't interpolate we just need the global maxima and minima
        # if self.interpolator is None:
        #     all_max = ecei_view.max()
        #     all_min = ecei_view.min()

        # # Find max an min after interpolation
        # else:
        #     all_max = -1.0
        #     all_min = 1.0
        #     for tslice in np.arange(ecei_view.data.shape[-1]):
        #         if self.interpolator is not None:
        #             frame_vals = (ecei_view.data[:, :, tslice], mask=ecei_view.bad_data)
        #             all_max = max(all_max, frame_vals.max())
        #             all_min = min(all_min, frame_vals.min())


        ### Alternative code
        ### Uses that interpolated values are in between old max and min.
        all_max = ecei_view.ecei_data[:, :, :][~ecei_view.bad_data].max()
        all_min = ecei_view.ecei_data[:, :, :][~ecei_view.bad_data].min()

        self.clevs = np.linspace(all_min, all_max, nlevs)

        

    def create_plot(self, ecei_view, time):
        tidx = ecei_view.tb.time_to_idx(time)
        print(f"Creating figure for time {time} - tidx {tidx}")

        if self.clevs is None:
            self.set_contour_levels(ecei_view)

        frame_vals = ecei_view.ecei_data[:, :, tidx]

        if self.interpolator is not None:
            frame_vals = self.interpolator(frame_vals, mask=ecei_view.bad_data)

        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.46, 0.75])
        ax_cb = fig.add_axes([0.7, 0.2, 0.05, 0.75])

        mappable = ax.contourf(self.rpos_arr, self.zpos_arr, frame_vals, levels=self.clevs)
        fig.colorbar(mappable, cax=ax_cb)


        return fig



# End of file plot_ecei.py