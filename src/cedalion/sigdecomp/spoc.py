import numpy as np
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from typing import Optional
import xarray as xr

class SPoC():
    """ Implements SPoC_lambda algorithm based on :cite:t:`BIBTEXLABEL`.

        Given a vector-valued time signal x(t) and a scalar target function z(t), 
        SPoC finds spatial filters W that maximize the covariance between
        the bandpower of the projected x signal, P(W.T @ x), and z. 
        Such a covariance defines the objective function of the problem, whose solution 
        can be formulated as the one for a generalized eigenvalue problem.
        The reconstructed sources s are given by the backward model s = W.T @ x.

        Assumptions:
            x(t) is of shape Nx x Nt, where Nx is the number of channels and Nt the 
            number of time points, and it is band-pass filtered in the frequency band 
            of interest. 
            z(e) is a standardize vector (zero mean and unit variance) of shape 1 x Ne, 
            where Ne < Nt is the number of "epochs". The latter represent labels 
            for intervals of the original time series. 
            Bandpower of the projected signal W.T @ x is then approximated by its 
            variance within epochs.
    """
    
    def __init__(self):

        # Spatial filters (initialized after calling the fit function)
        self.W = None

    def fit(self,
            x: cdt.NDTimeSeries,
            z: xr.DataArray, 
            only_max: bool = True) -> np.ndarray:
        """ Fit the model on the (x, z) dataset.

        Solve the generalized eigenvalue problem and store the trained spatial filters W
        as a local state of the class.
        
        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Temporal signal 
            of shape Nx x Nt.
            z (:class:`DataArray`, (time)): Target (scalar) function
            of shape 1 x Ne, with Ne < Nt. 
            only_max: If True, it returns only the filter corresponding to 
                maximum eigenvalue/covariance. If False, it returns all Nx components.
        
        Returns:
            scores: Array of Nx eigenvalues. The latter also coincide with 
                the corresponding covariances between P(W.T @ x) and z. 
                If only_max = True, eig_values is a single element, corresponding 
                to the maximum covariance. 
        """
        
        # Catch incompatible lengths
        Ne = len(z.time)
        Nt = len(x.time)
        Nx = len(x.channel)
        if Nt <= Ne:
            raise ValueError("x should have more time points than z")
        
        # Standardize z
        z = standardize(z, dim='time')
        
        # Split signal into epochs and build matrices for the eigenvalue problem
        x_epochs = x.groupby_bins(group='time', bins=Ne)
        Cxxe = np.stack([np.cov(xe) for _, xe in x_epochs])  # Epoch-wise cov. matrix
        Cxx = Cxxe.mean(axis=0)
        Cxxz = (Cxxe * z.values.reshape(-1, 1, 1)).mean(axis=0)
        
        # Select only the maximum-covariance component if only_max
        subset = [Nx - 1, Nx - 1] if only_max else None

        # Solve generalized eigenvalue problem
        scores, W = eigh(Cxxz, Cxx, eigvals_only=False, subset_by_index=subset)
        # Update state
        self.W = xr.DataArray(W, coords={'channel': x.channel, 
                                         'component': [f'S{i}' 
                                                       for i in range(len(scores))]})
        
        return scores
    
    def transform(self,
                  x: cdt.NDTimeSeries, 
                  only_component: bool = False,
                  Ne: Optional[int] = None) -> xr.DataArray | tuple[xr.DataArray, 
                                                                    xr.DataArray]:
        """ Apply backward model to x to build reconstructed sources.

        Get reconstructed sources s by projecting x along the spatial filtes.
        If only_component = False, also estimate epoch-wise bandpower of the
        components via the per-epoch variance. 

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Temporal signal
                of shape Nx x Nt.
            only_component: Wether to return only the reconstructed sources
                or also the epoch-wise bandpower.
            Ne: Number of epochs along which to estimate the bandpower.

        Returns:
            s: Reconstructed sources (W.T @ x).
            s_power: standardized epoch-wise bandpower of s (Var(W.T @ x)).
        """
        
        # Build reconstructed sources from projection
        s = self.W.T @ x

        if only_component:
            return s
        
        else:
            # Split into epochs, estimate bandpower, and standardize
            s_epochs = s.groupby_bins(group='time', bins=Ne)
            s_power = standardize(s_epochs.var(), dim='time_bins')

            return s, s_power



class mSPoC():
    """ Implements mSPoC algorithm based on :cite:t:`BIBTEXLABEL`.

    TODO: Add extense description.
    
    Args:
        n_lags (int): Number of time lags for temporal embedding.
        n_components (int): Number of components/filter/eigenvalues 
            the algorithm will find.
    """

    def __init__(self, n_lags: int = 5, n_components: int = 1):

        self.n_lags = n_lags
        self.n_components = n_components  # TODO: Add multiple components option

        # Time lags, including t_0=0 (no lag)
        self.t_lags = np.arange(0, self.n_lags)

        # Spatial and temporal filters (initialized after calling the fit function)
        self.wx = None
        self.wy = None
        self.wt = None

    @cdc.validate_schemas
    def fit(self, 
            x: cdt.NDTimeSeries, 
            y: cdt.NDTimeSeries, 
            N_restarts: int =10, 
            max_iter: int = 200, 
            tol: float = 1e-5) -> float:
        """Train mSPoC model on the x, y dataset

        Implement the pseudo-code of Algorithm 1 of :cite:t:`BIBTEXLABEL` 
        for a single component pair. After training, the filter attributes
        wx, wy, and wt are updated.

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                first modality
            y (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                second modality.
            N_restarts (int): Number of times the algorithm is repeated.
            max_ter (int): Maximum number of iterations.
            tol (float): Tolerance value used for convergence criterion 
                when comparing correlations of consecutive runs.
        Returns:
            corr_best (float): Best correlation achieved among all repetitions
            
        """

        # TODO: Use the xarray datatype instead of calling .data from the very beginning
        Nx = len(x.channel)
        Nt = len(x.time)
        Ne = len(y.time)
        e_len = Nt//Ne

        # Remove first epochs so its dimensions match bandpower Phi_x later
        y = y.values[:, self.n_lags - 1:]

        # Split signal into epochs
        x_epochs = self.split_epochs(x.values, Ne, e_len)
        # Epoch-wise covariance matrix, its temporal embedding and mean
        Cxxe = np.stack([np.cov(x_e) for x_e in x_epochs])
        tCxxe = self.temporal_embedding(Cxxe)
        Cxx = Cxxe.mean(axis=0)
        # To keep track of best model
        corr_best = 0.0
            
        for i in range(N_restarts):
            
            # Initialize random filter
            wx = np.random.normal(0, 1, [Nx, 1])
            # Epoch-wise bandpower with temporal embedding
            Phi_x = self.get_bandpower(wx, Cxxe)
            # For convergence condition
            converged = False
            corr_old = 0.0

            for i in range(max_iter):
                # Apply CCA to get wt and wy
                wt, wy = self.apply_cca(Phi_x.T, y.T)
                # Apply temporal filter to tCxxe
                hCxxe = (wt.reshape(1, -1, 1, 1) * tCxxe).sum(axis=1)
                # Reconstructed y-source
                sy = wy.T.dot(y)
                # Build matrix for SPoC algorithm 
                hsy = (sy.reshape(-1, 1, 1) * hCxxe).mean(axis=0)
                # Solve generalized eigenvalue problem (SPoC)
                _, wx = eigh(hsy, Cxx, eigvals_only=False, 
                             subset_by_index=[Nx - 1, Nx - 1])
                # Epoch-wise bandpower with temporal embedding
                Phi_x = self.get_bandpower(wx, Cxxe)
                # Compute correlation
                corr = np.corrcoef(wt.T.dot(Phi_x), sy)[0, 1]
                # Check for convergence
                if np.abs(corr - corr_old) < tol:
                    converged = True
                    break
                else:
                    corr_old = corr

            # Check for convergence
            if converged:
                if corr > corr_best:  # Save filters for best model
                    self.wx = wx
                    self.wy = wy
                    self.wt = wt
                    corr_best = corr
            else:
                print("mSPoC did not converged! (reached maximum number of iterations)")

        return corr_best
    
    def transform(self, 
                  x: cdt.NDTimeSeries, 
                  y: cdt.NDTimeSeries) -> tuple:
        """ Get reconstructed sources of the x and y dataset.

        The x component is constructed by computing the bandpower of the x projection 
        along wx, and then applying a liner temporal filter using wt. 
        The y component is constructed as the linear projection of y along wy.

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                first modality
            y (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                second modality.

        Return:
            sx (ndarray): (Bandpower of) reconstructed x source.
            sy (ndarray): Reconstructed y source.

        """

        Nt = len(x.time)
        Ne = len(y.time)
        e_len = Nt//Ne

        # Remove first epochs so its dimensions match bandpower Phi_x later
        y = y.data[:, self.n_lags - 1:]

        # Split signal into epochs
        x_epochs = self.split_epochs(x.values, Ne, e_len)
        # Epoch-wise covariance matrix
        Cxxe = np.stack([np.cov(x_e) for x_e in x_epochs])
        # Epoch-wise bandpower with temporal embedding
        Phi_x = self.get_bandpower(self.wx, Cxxe)
        # Reconstructed sources
        sx = self.wt.T.dot(Phi_x)
        sy = self.wy.T.dot(y)

        return sx, sy


    
    def temporal_embedding(self, v):
        """Build temporal embedding of v
        
        Stack copies of v shifted by the time lags. It assumes time direction 
        corresponds to index 0.
        """

        v_embedding = np.stack([
            v[e - self.t_lags] for e in range(self.n_lags - 1, v.shape[0])
            ])

        return v_embedding


    def get_bandpower(self, w, C):
        """Compute bandpower with temporal embedding

        TODO: Add description.
        """

        # Epoch-wise power
        phi_x = np.stack([w.T.dot(c).dot(w) for c in C]).squeeze()
        # Temporal embedding
        Phi_x = self.temporal_embedding(phi_x).T

        return Phi_x
    
    @staticmethod
    def apply_cca(a, b):
        """Initialize and fit CCA to the a, b pair

        TODO: Add description.
        """

        cca = CCA(n_components=1).fit(a, b)
        wa = cca.x_weights_
        wb = cca.y_weights_

        return wa, wb
        
    @staticmethod
    def split_epochs(x, Ne, e_len):
        """Split a signal x into Ne epochs of length e_len (index)
        TODO: Add description.
        """

        return np.stack([x[:, i*e_len:(i+1)*e_len] for i in range(Ne)])
        

def standardize(x: cdt.NDTimeSeriesSchema, dim: str = 'time'):
    """Standardize x along dimension dim.
    """

    x_standard = (x - x.mean(dim))/x.std(dim)

    return x_standard