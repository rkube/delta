#Encoding: UTF-8 -*-

"""
Author: Minjun Choi (original), Ralph Kube

Contains helper function for working with the ECEI diagnostic.
These are just the member functions from kstarecei.py, copied here
so that we don't need to instantiate an kstarecei object."""

import numpy as np

def get_abcd(channel, LensFocus, LensZoom, Rinit, new_H=True):
        """Input:
        ch: channel
        LensZoom: float
        LensFocus: float
        Rinit: ???
        new_H: If true, use new values for H-dev, shot > 12957

        Returns:
        ========
        ABCD: The ABCD matrix
        """

        # ABCD matrix
        if channel.dev == 'L':
            sp = 3350 - Rinit*1000  # [m] -> [mm]
            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-LensZoom],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,LensZoom],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]])).dot(
                   np.array([[1,710-LensFocus+140],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1270),1.52]])).dot(
                   np.array([[1,90],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1270*1.52),1/1.52]])).dot(
                   np.array([[1,539+35+LensFocus],[0,1]]))

        elif channel.dev == 'H':
            sp = 3350 - Rinit*1000
            abcd = np.array([[1,250+sp],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-730),1.52]])).dot(
                   np.array([[1,135],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(2700*1.52),1/1.52]])).dot(
                   np.array([[1,1265-LensZoom],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1100,1.52]])).dot(
                   np.array([[1,40],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1100*1.52),1/1.52]])).dot(
                   np.array([[1,LensZoom],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,65],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(800*1.52),1/1.52]]))

            if new_H:
                abcd = abcd.dot(
                   np.array([[1,520-LensFocus+590-9.2],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1100),1.52]])).dot(
                   np.array([[1,88.4],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1100*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+LensFocus-9.2],[0,1]]))
            else:
                abcd = abcd.dot(
                   np.array([[1,520-LensFocus+590],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1400),1.52]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1400*1.52),1/1.52]])).dot(
                   np.array([[1,446+35+LensFocus],[0,1]]))

        elif channel.dev == 'G':
            sp = 3150 - Rinit*1000
            abcd = np.array([[1,1350-LensZoom+sp],[0,1]]).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,100],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(900*1.545),1/1.545]])).dot(
                   np.array([[1,1430-LensFocus+660+LensZoom+470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,70],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,LensFocus-470],[0,1]])).dot(
                   np.array([[1,0],[0,1.545]])).dot(
                   np.array([[1,80],[0,1]])).dot(
                   np.array([[1,0],[(1-1.545)/(800*1.545),1/1.545]])).dot(
                   np.array([[1,390],[0,1]]))

        elif channel.dev == 'GT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-LensZoom)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-LensZoom)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-LensFocus],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-LensFocus)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))

        elif channel.dev == 'GR':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+(1954-LensZoom)],[0,1]]).dot(
                   np.array([[1,0],[(1.52-1)/(-1000),1.52]])).dot(
                   np.array([[1,160],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1000*1.52),1/1.52]])).dot(
                   np.array([[1,2280-(1954+160-LensZoom)],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/1000,1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4288-(2280+20)-LensFocus],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1200),1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1200*1.52),1/1.52]])).dot(
                   np.array([[1,4520-(4288+140-LensFocus)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,30],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,4940-(4520+30)],[0,1]]))

        elif channel.dev == 'HT':
            sp = 2300 - Rinit*1000
            abcd = np.array([[1,sp+2586],[0,1]]).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,140],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(770*1.52),1/1.52]])).dot(
                   np.array([[1,4929-(2586+140)-LensZoom],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(1200),1.52]])).dot(
                   np.array([[1,20],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(-1200*1.52),1/1.52]])).dot(
                   np.array([[1,5919-(4929+20-LensZoom)-LensFocus],[0,1]])).dot(
                   np.array([[1,0],[(1.52-1)/(-1300),1.52]])).dot(
                   np.array([[1,130],[0,1]])).dot(
                   np.array([[1,0],[(1-1.52)/(1300*1.52),1/1.52]])).dot(
                   np.array([[1,6489-(5919+130-LensFocus)],[0,1]])).dot(
                   np.array([[1,0],[0,1.52]])).dot(
                   np.array([[1,25.62],[0,1]])).dot(
                   np.array([[1,0],[0,1/1.52]])).dot(
                   np.array([[1,7094.62-(6489+25.62)],[0,1]]))

        return abcd


def beam_path(ch, LensFocus, LensZoom, rpos):
    """Calculates the ray vertical position and angle at rpos [m] ray starting
    from the array box position.

    Input:
    ======
    ch, channel.
    LensFocus, float
    LensZoom: float
    rpos: float, radial position of the channel view, in meters
    ch_v: int, number of vertical channel
    """

    abcd = get_abcd(ch, LensFocus, LensZoom, rpos)

    # vertical position from the reference axis (vertical center of all lens, z=0 line) at ECEI array box
    zz = (np.arange(24, 0, -1) - 12.5) * 14  # [mm]
    # angle against the reference axis at ECEI array box
    aa = np.zeros(np.size(zz))

    # vertical posistion and angle at rpos
    za = np.dot(abcd, [zz, aa])
    zpos = za[0][ch.ch_v - 1] * 1e-3  # zpos [m]
    apos = za[1][ch.ch_v - 1]  # angle [rad] positive means the (z+) up-directed (divering from array to plasma)

    print(f"   LensFocus={LensFocus} LensZoom={LensZoom} rpos={rpos:5.3f} abcd={abcd}")

    return zpos, apos

def channel_position(ch, ecei_cfg):
    """Calculates the position of a channel in configuration space

    Input:
    ======
    ch, channel, The channel whos position we want to calculate
    ecei_cfg, dict: Parameters of the ECEi diagnostic.
    """
    
    me = 9.1e-31             # electron mass, in kg
    e = 1.602e-19            # charge, in C
    mu0 = 4. * np.pi * 1e-7  # permeability
    ttn = 56*16              # total TF coil turns

    # Unpack ecei_cfg
    TFcurrent = ecei_cfg["TFcurrent"] # Instead of multiplying by 1e3, we put this in the config file
    LoFreq = ecei_cfg["LoFreq"]
    LensFocus = ecei_cfg["LensFocus"]
    LensZoom = ecei_cfg["LensZoom"]
    # Set hn, depending on mode. If mode is undefined, set X-mode as default.
    try:
        if ecei_cfg["Mode"] == 'O':
            hn = 1
        elif ecei_cfg["Mode"] == 'X':
            hn = 2

    except KeyError as k:
        print("ecei_cfg: key {0:s} not found. Defaulting to 2nd X-mode".format(k.__str__()))
        ecei_cfg["Mode"] = 'X'
        hn = 2
    
    print(f"channel_position: ch_v={ch.ch_v}, ch_h={ch.ch_h}")

    rpos = hn * e * mu0 * ttn * TFcurrent /\
                (4. * np.pi * np.pi * me * ((ch.ch_h - 1) * 0.9 + 2.6 + LoFreq) * 1e9)
    
    zpos, apos = beam_path(ch, LensFocus, LensZoom, rpos)
    print(f"       rpos={rpos:5.3f}, zpos={zpos:5.3f}, apos={apos:5.3f}")
    return (rpos, zpos, apos)


# End of file ecei_helper.py