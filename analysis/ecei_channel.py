# -*- Encoding: UTF-8 -*-

import numpy as np
import warnings
import h5py
from numpy.compat.py3k import os_fspath
from channels import channel
import time


class timebase():
    def __init__(self, t_start, t_end, f_sample):
        """
        Defines a time base for ECEI channel data
        Parameters
        ----------
            t_trigger: float, 
            t_offset: float,
            f_sample: float, sampling frequency of the ECEI data
        """
        # Assume that 0 <= t_start < t_end 
        assert(t_end >= 0.0)
        assert(t_start < t_end)
        assert(f_sample >= 0.0)
        self.t_start = t_start
        self.t_end = t_end
        self.f_sample = f_sample
        self.dt = 1. / self.f_sample

        self.num_samples = int((self.t_end - self.t_start) / self.dt)


    def time_to_idx(self, t0):
        """
        Given a timestamp, returns the index where the timebase is closest.

        Parameters:
        t0: float - Time stamps that we want to calculate the index for
        """
        # Ensure that the time we are looking for is inside the domain
        assert(t0 >= self.t_start)
        assert(t0 <= self.t_end)

        fulltime = self.get_fulltime()
        idx = np.argmin(np.abs(fulltime - t0))

        return idx

        
    def get_fulltime(self):
        """
        Returns an array with the full timebase

        """

        return np.arange(self.t_start, self.t_end, self.dt)



class ecei_channel():
    """Defines data as an ECEI channel"""

    def __init__(self, data, tb, channel, t_offset=None, t_crop=None):
        """
        Parameters:
        -----------
        data: ndarray, float - Raw Voltages from the ECEI diagnostic
        tb: timebase - Timebase object for the raw voltages
        channel: channel object - Describes the channel
        t_offset: tuple (t_n0, t_n1) - Tuple that defines the time interval where a signal reference value is calculated. If None,
                                     raw values will be used.
        t_crop: tuple (t_c0, t_c1) - Defines the time interval where the data is cropped to. If None, data will not
                                     be cropped

        """

        # Make sure that the timebase is appropriate for the data
        assert(np.max(data.shape) == tb.num_samples)
        self.data = data * 1e-4
        self.tb_raw = tb
        self.channel = channel

        self.is_cropped = False
        self.is_normalized = False

        if t_offset is not None:
            # Calculate the signal offset
            self.calculate_offsets(t_offset)

            if t_crop is not None:
                self.crop_data(t_crop)
                self.is_cropped = True

            # Subtract signal offset after signal has been cropped
            self.data = (self.data - self.offlev) 

        else:
            if t_crop is not None:
                self.crop_data(t_crop)
                self.is_cropped = True

        self.siglev = np.median(self.data)
        self.sigstd = self.data.std()

        # After signal is shifted and cropped we normalize the signal
        self.data = self.data / self.data.mean() - 1.0

        print("all good")


    def calculate_offsets(self, t_offset):
        """Calculate mean and standard deviation from un-normalized channel data
        Parameters:
        -----------
        t_norm: t_n0, t_n1) - Tuple that defines the time interval where the data is normalized to
        """

        if self.is_normalized == False:
            # Calculate normalization constants. See fluctana.py, line 118ff
            idx_norm = [self.tb_raw.time_to_idx(t) for t in t_offset]

            offset_interval = self.data[idx_norm[0]:idx_norm[1]]
            print(f"Calculating offsets at {idx_norm[0]:d}:{idx_norm[1]:d}")
            self.offlev = np.median(offset_interval)
            self.offstd = offset_interval.std()


    def calculate_sigstats(self):
        """Calculate signal statistics. Before normalization.
        """

        assert(self.is_normalized == False)
        self.siglev = np.median(self.data)
        self.sigstd = self.data.std()

    
    def crop_data(self, t_crop):
        """Crops the data to the interval defined by t_crop.

        Input:
        ======
        t_crop: Tuple (t0, t1), where t0 and t1 correspond to the timebase passed into __init__
        """
        if self.is_cropped == False:
            idx_crop = [self.tb_raw.time_to_idx(t) for t in t_crop]
            print("Cropping data using ", idx_crop)
            self.data = self.data[idx_crop[0]:idx_crop[1]]
            print(f"data[0] = {self.data[0]}, data[-1] = {self.data[-1]}")


    def position(self):
        # Returns the R,Z position of the channel
        pass

    def channel_name(self):
        # Returns the channel name
        pass

    def get_timebase(self):
        """Generates and returns a timebase object for the channel data"""

        pass

    def check_signal_level(self):
        """
        Returns True if the signal level is acceptable
        Returns False if the signal level is bad
        """
        
        # Signal level is defined as the median, see fluctana.py line

        if self.siglev > 0.01:
            ref = 100. * self.offstd / self.siglev
        else:
            ref = 100.

        if ref > 30.0:
            warnings.warn(f"LOW signal level channel {self.channel:s}, ref = {ref:4.1f}, siglevel = {self.siglev:4.1f} V")
            return False

        return True


    def check_bottom_sat(self):
        """
        Check bottom saturation.
        Good saturation: Return True if bottom saturation is above 0.001
        Bad saturation: Return False if bottom saturation is below 0.001
        """

        if self.offstd < 0.0001:
            warning.warn(f"SAT offstd data channel {self.channel:s}, offstd = {self.offstd:g}%, offlevel = {self.offlev:g} V")
            return False

        return True


    def check_top_sat(self):
        """
        Check top saturation.
        Good saturation: Return True if top saturation is above 0.001
        Bad saturation: Return False if top saturation is below 0.001
        """

        if self.sigstd < 0.001:
            warning.warn(f"SAT sigstd data channel {self.channel:s}, sigstd = {self.sigstd:g}%, offlevel = {self.offlev:g} V")
            return False

        return True


class ecei_view():
    """Defines the view of an ECEI. This extends ecei_channel to the entire diagnostic
    
    Parameters:
    -----------
    datafilename: string, filename to the HDF5 file
    tb: timebase - Timebase object for the raw voltages
    dev: device name
    t_offset: tuple (t_n0, t_n1) - Tuple that defines the time interval where a signal reference value is calculated. If None,
                                    raw values will be used.
    t_crop: tuple (t_c0, t_c1) - Defines the time interval where the data is cropped to. If None, data will not
                                    be cropped
    
    
    """

    def __init__(self, datafilename, tb, dev, t_offset=(-0.099, -0.089), t_crop=(1.0, 1.1), num_v=24, num_h=8):

        # Number of vertical and horizontal channels
        self.num_v = 24
        self.num_h = 8
        # Infer number of samples in the cropped interval
        idx_offset = [tb.time_to_idx(t) for t in t_offset]
        idx_crop = [tb.time_to_idx(t) for t in t_crop]
        self.num_samples = idx_crop[1] - idx_crop[0]

        # Use float32 since data is generated from 16bit integers
        self.ecei_data = np.zeros([self.num_v, self.num_h, self.num_samples], dtype=np.float32)
        # Marks data the we ignore for plotting etc.
        self.bad_data = np.zeros([self.num_v, self.num_h], dtype=bool)

        # Offset level
        self.offlev = np.zeros([self.num_v, self.num_h], dtype=np.float32)
        # Offset standard deviation
        self.offstd = np.zeros([self.num_v, self.num_h], dtype=np.float32)
        # Signal level
        self.siglev = np.zeros([self.num_v, self.num_h], dtype=np.float32)
        # Signal standard deviation
        self.sigstd = np.zeros([self.num_v, self.num_h], dtype=np.float32)

        tic = time.perf_counter()
        # Load data from HDF file
        with h5py.File(datafilename, "r") as df:   
            print("Trigger time: ", df['ECEI'].attrs['TriggerTime'])
            for ch_idx in range(192):
                ch_v, ch_h = np.mod(ch_idx, 24), ch_idx // 24
                ch_str = f"/ECEI/ECEI_{dev}{(ch_v + 1):02d}{(ch_h + 1):02d}/Voltage"
                
                # Calculate the start-of-shot offset
                self.offlev[ch_v, ch_h] = np.median(df[ch_str][idx_offset[0]:idx_offset[1]]) * 1e-4
                self.offstd[ch_v, ch_h] = df[ch_str][idx_offset[0]:idx_offset[1]].std() * 1e-4
                
                tmp = df[ch_str][idx_crop[0]:idx_crop[1]] * 1e-4  - self.offlev[ch_v, ch_h]

                self.siglev[ch_v, ch_h] = np.median(tmp)
                self.sigstd = tmp.std()
                self.ecei_data[ch_v, ch_h, :] = tmp / tmp.mean() - 1.0

        toc = time.perf_counter()

        print(f"Loading data took {(toc - tic):4.2f}s")

        self.tb = timebase(t_crop[0], t_crop[1], tb.f_sample)

        self.mark_bad_channels(verbose=True)


    def mark_bad_channels(self, verbose=False):
        """Mark bad channels. These are channels with either
        * Low signal level: std(offset) / siglev > 0.3
        * Saturated signal data(bottom saturation): std(offset) < 0.001
        * Saturated offset data(top saturation): std(signal) < 0.001

        """

        # Check for low signal level
        ref = 100. * self.offstd / self.siglev
        ref[self.siglev < 0.01] = 100

        if verbose:
            for item in np.argwhere(ref > 30.0):
                print(f"LOW SIGNAL: channel({item[0] + 1:d},{item[1] + 1:d}): {ref[tuple(item)]*1e2:4.2f}")    
        self.bad_data[ref > 30.0] = True

        # Mark bottom saturated channels
        self.bad_data[self.offstd < 1e-3] = True
        if verbose:
            for item in np.argwhere(self.offstd < 1e-3):
                os = self.offstd[tuple(item)]
                ol = self.offlev[tuple(item)]
                print(f"SAT offset data channel ({item[0] + 1:d}, {item[1] + 1:d}) offstd = {os} offlevel = {ol}")

        # Mark top saturated channels
        self.bad_data[self.sigstd < 1e-3] = True
        if verbose:
            for item in np.argwhere(self.sigstd < 1e-3):
                os = self.offstd[tuple(item)]
                ol = self.offlev[tuple(item)]
                print(f"SAT signal data channel ({item[0] + 1:d}, {item[1] + 1:d}) offstd = {os} offlevel = {ol}")


# End of file ecei_channel.py