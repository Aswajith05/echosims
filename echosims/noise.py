import sys
import os
from pathlib import Path
import numpy as np
import healpy as hp

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from echosims.utils import EchoInstrument

class Noise(EchoInstrument):

    def __init__(self):
        super().__init__()
        self.EchoInstrument = EchoInstrument
        pass

    def TQU(self, band, nside):

        npix = hp.nside2npix(nside)
        pix_area = hp.nside2pixarea(nside, degrees = True) # Unit - Degrees^2
        pix_len = np.sqrt(pix_area) # Unit - Degrees

        # Temperature Noise Map
        sigma_T = self.get_temperature_sensitivity(band)/60 # Unit conversion from muKarmin to muKdegree
        std_dev_T = sigma_T/pix_len
        T_noise = np.random.normal(0, std_dev_T, npix)

        # Q Noise Map
        sigma_Q = self.get_polarization_sensitivity(band)/60 # Unit conversion from muKarmin to muKdegree
        std_dev_Q = sigma_Q/pix_len
        Q_noise = np.random.normal(0, std_dev_Q, npix)

        # U Noise Map
        sigma_U = self.get_polarization_sensitivity(band)/60 # Unit conversion from muKarmin to muKdegree
        std_dev_U = sigma_U/pix_len
        U_noise = np.random.normal(0, std_dev_U, npix)

        return (T_noise, Q_noise, U_noise)
        