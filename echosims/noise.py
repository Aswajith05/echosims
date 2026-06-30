import sys
from pathlib import Path
import numpy as np
import healpy as hp

repo_root = Path(__file__).resolve().parent.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from echosims.utils import EchoInstrument
    """
    Generate T, Q, U white-noise maps for a given instrumental band.
    """

    def __init__(self, nside, seed = 33333):
        super().__init__()

        self.nside = nside
        self.lmax = 3 * nside - 1
        self.SEED = seed


    def TQU(self, band, idx):
        """
        Generate one TQU noise realization for the given band.

        Parameters
        ----------
        band : int
            Frequency band label used by EchoInstrument.

        idx : int
            Iteration index.

        Returns
        -------
        TQU_maps : numpy.ndarray
            Array of shape (3, npix), ordered as T, Q, U.
        """

        # Generating Noise Spectra
        sigma_T = self.get_temperature_sensitivity(band) # Unit - muK.arcmin
        sigma_P = self.get_polarization_sensitivity(band) # Unit - muK.arcmin
        
        # Convert muK-arcmin sensitivity to white-noise power spectrum.
        # The resulting n_l has units of muK^2 sr.
        nl_TT = np.full(self.lmax + 1, (np.radians(sigma_T/60))**2) 
        nl_PP = np.full(self.lmax + 1, (np.radians(sigma_P/60))**2) # This serves as both EE and BB Spectra 

        nl_TE = np.zeros(self.lmax + 1)

        np.random.seed(self.SEED + idx)
        
        TQU_maps = hp.synfast(
            [nl_TT, nl_PP, nl_PP, nl_TE], 
            nside = self.nside,
            lmax = self.lmax,
            pol = True,
            new = True,
            verbose = False
        )

        return TQU_maps
