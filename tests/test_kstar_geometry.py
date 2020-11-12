# -*- Encoding: UTF-8 -*-

"""Verfify calculated ECEI channel positions against fluctana."""


class kstarecei_mockup():
    """Mocks behaviour of kstarecei, but without data dependencies."""
    def __init__(self, ch_h, ch_v, dev):
        """Initializes.

        Args:
            ch_h (int):
                Horizontal channel
            ch_v (int):
                Vertical channel
            dev (char):
                Device name

        Returns:
            None
        """
        # Initialize with channel information
        self.ch_h = ch_h
        self.ch_v = ch_v
        self.dev = dev
        # Use values for shot 18431 whenever possible
        self.lofreq = 81
        self.tfcurrent = 23000.0
        self.sf = 80 # LensFocus
        self.sz = 340 # LensZoom

    def channel_position(self):
        """From fluctana/kstarecei.py. Calculates channel position."""
        import numpy as np
        # get self.rpos, self.zpos, self.apos
        # NEED corrections using syndia

        me = 9.1e-31            # electron mass
        e = 1.602e-19           # charge
        mu0 = 4 * np.pi * 1e-7  # permeability
        ttn = 56 * 16           # total TF coil turns
        harmonic = 1            # O-mode in shot 18431

        rpos = harmonic * e * mu0 * ttn * self.tfcurrent / ((2 * np.pi)**2 * me *
                                                            ((self.ch_h - 1) * 0.9 +
                                                            2.6 + self.lofreq) * 1e9)

        zpos, apos = self.beam_path(rpos, self.ch_v)

        return (rpos, zpos, apos)
        #  cnum = len(self.clist)
        #  self.rpos = np.zeros(cnum)  # R [m] of each channel
        #  self.zpos = np.zeros(cnum)  # z [m] of each channel
        #  self.apos = np.zeros(cnum)  # angle [rad] of each channel
        #  for c in range(0, cnum):
        #      vn = int(self.clist[c][(self.cnidx1):(self.cnidx1+2)])
        #      fn = int(self.clist[c][(self.cnidx1+2):(self.cnidx1+4)])
        #      # assume cold resonance with Bt ~ 1/R
        #      self.rpos[c] = self.hn*e*mu0*ttn*self.itf/((2*np.pi)**2*me*((fn - 1)*0.9 + 2.6 + self.lo)*1e9)

        #      # get vertical position and angle at rpos
        #      self.zpos[c], self.apos[c] = self.beam_path(self.rpos[c], vn)   

    def beam_path(self, rpos, vn):
        """From fluctana. Calcuates beampath."""
        import numpy as np
        # IN : shot, device name, R posistion [m], vertical channel number
        # OUT : a ray vertical position and angle at rpos [m] [rad]
        # this will find a ray vertical position and angle at rpos [m]
        # ray starting from the array box posistion

        abcd = self.get_abcd(self.sf, self.sz, rpos)

        # vertical position from the reference axis
        # (vertical center of all lens, z=0 line) at ECEI array box
        zz = (np.arange(24, 0, -1) - 12.5) * 14  # [mm]
        # angle against the reference axis at ECEI array box
        aa = np.zeros(np.size(zz))

        # vertical posistion and angle at rpos
        za = np.dot(abcd, [zz, aa])
        zpos = za[0][vn - 1] / 1000  # zpos [m]
        apos = za[1][vn - 1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

        return zpos, apos

    def get_abcd(self, sf, sz, Rinit):
        import numpy as np
        """Calculate the ABCD matrix."""
        if self.dev == 'L':
            sp = 3350 - Rinit*1000  # [m] -> [mm]
            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,sz],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]])).dot(
                   np.array([[1,710-sf+140],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1270),1.52]])).dot(
                   np.array([[1,90],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1270*1.52),1/1.52]])).dot(
                  np.array([[1,539+35+sf],[0,1]]))
        elif self.dev == 'H':
            sp = 3350 - Rinit*1000
            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,sz],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]]))
            if self.shot > 12297:  # since 2015 campaign
                abcd = abcd.dot(
                   np.array([[1,520-sf+590-9.2],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1100),1.52]])).dot(
                   np.array([[1,88.4],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1100*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+sf-9.2],[0,1]]))
            else:
                abcd = abcd.dot(
                   np.array([[1,520-sf+590],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1400),1.52]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1400*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+sf],[0,1]]))
        elif self.dev == 'G':
            sp = 3150 - Rinit*1000
            abcd = np.array([[1,1350-sz+sp],[0,1]]).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,100],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(900*1.545),1/1.545]])).dot(
                   np.array([[1,1430-sf+660+sz+470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,sf-470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,80],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,390],[0,1]]))
        elif self.dev == 'GT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'GR':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-sz)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-sz)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))
        elif self.dev == 'HT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+2586],[0,1]]).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(770*1.52),1/1.52]])).dot(
                   np.array([[1,4929-(2586+140)-sz],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(1200),1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1200*1.52),1/1.52]])).dot(
                   np.array([[1,5919-(4929+20-sz)-sf],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1300),1.52]])).dot(
                   np.array([[1,130],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1300*1.52),1/1.52]])).dot(
                   np.array([[1,6489-(5919+130-sf)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,25.62],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,7094.62-(6489+25.62)],[0,1]]))

        return abcd


def test_ecei_channel_geom():
    """Verify calculated ECEI channel positions are the same as in fluctana."""
    import sys
    import os
    sys.path.append(os.path.abspath('delta'))
    import numpy as np
    # Import packages as delta.... so that we can run pytest as 
    from delta.data_models.channels_2d import channel_2d
    from delta.data_models.kstar_ecei import get_geometry

    # Pick a random channel
    ch_h = np.random.randint(1, 9)
    ch_v = np.random.randint(1, 25)
    # ch_str = f"ECEI_L{ch_v:02d}{ch_h:02d}"
    ch_2d = channel_2d(ch_v, ch_h, 24, 8, order="horizontal")

    # Manually provide ECEI configuration dictionary for Delta
    cfg_diagnostic = {"name": "kstarecei", "shotnr": 18431,
                      "datasource": {
                          "source_file": "/global/cscratch1/sd/rkube/KSTAR/kstar_streaming/018431/ECEI.018431.LFS.h5",
                          "chunk_size": 10000,
                          "num_chunks": 500,
                          "channel_range": ["L0101-2408"],
                          "datatype": "float"},
                      "parameters": {
                          "Device": "L",
                          "TriggerTime": [-0.12, 61.2, 60],
                          "t_norm": [-0.119, -0.109],
                          "SampleRate": 500,
                          "TFcurrent": 23000.0,
                          "Mode": "O",
                          "LoFreq": 81,
                          "LensFocus": 80,
                          "LensZoom": 340}}

    # Calcuate channel position using FluctAna and Delta
    K = kstarecei_mockup(ch_h, ch_v, "L")
    pos_true = K.channel_position()
    rpos_arr, zpos_arr, apos_arr = get_geometry(cfg_diagnostic["parameters"])
    # print("Re-factored:", ch_2d.get_idx())
    # print(f"rpos = {rpos_arr[ch_2d.get_idx()]}, zpos = {zpos_arr[ch_2d.get_idx()]}, apos = {apos_arr[ch_2d.get_idx()]}")
    pos_delta = np.array([rpos_arr[ch_2d.get_idx()], zpos_arr[ch_2d.get_idx()], apos_arr[ch_2d.get_idx()]])

    assert(np.linalg.norm(pos_true - pos_delta) < 1e-8)

# End of file test_kstar_ecei_helpers.py
