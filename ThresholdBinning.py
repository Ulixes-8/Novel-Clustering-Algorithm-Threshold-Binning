import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.base import BaseEstimator, ClusterMixin

class ThresholdBinning(BaseEstimator, ClusterMixin):
    """
    A custom clustering algorithm based on threshold binning.

    Parameters
    ----------
    max_bins : int, default=5
        The maximum number of bins to create.
        
    threshold : float, default=0.1
        The distance threshold for assigning data points to bins.
        
    min_bin_size : int, default=5
        Minimum number of samples required in a bin.
        
    verbose : bool, default=False
        Verbosity flag.
        
    Attributes
    ----------
    bins : dict
        Dictionary containing bins and their data points.
        
    centroids : dict
        Dictionary containing centroids of each bin.
    """

    def __init__(self, max_bins=5, threshold=0.1, min_bin_size=None, verbose=False):
        self.max_bins = max_bins
        self.threshold = threshold
        self.min_bin_size = min_bin_size
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the model.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The input data.
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._threshold_binning(X)
        return self

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict cluster indices.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The input data.
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns
        -------
        labels : array [n_samples]
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self._assign_labels(X)

    def _normalize_rows(self, df):
        return df.apply(lambda row: row / row.sum(), axis=1)

    def _calculate_centroid(self, bin_data):
        if not bin_data:
            return None
        return np.mean(bin_data, axis=0).tolist()

    def _threshold_binning(self, X):
        X = self._normalize_rows(X)
        self.bins = {0: [X.iloc[0].tolist()]}
        self.centroids = {0: self._calculate_centroid(self.bins[0])}

        for i in range(1, len(X)):
            self._assign_to_bin_or_create_new(X.iloc[i].tolist())

    def _assign_to_bin_or_create_new(self, row):
        assigned_bin, min_distance = self._find_closest_bin(row)
        if assigned_bin is not None:
            self.bins[assigned_bin].append(row)
            self.centroids[assigned_bin] = self._calculate_centroid(self.bins[assigned_bin])
        elif len(self.bins) < self.max_bins:
            new_bin_key = max(self.bins.keys()) + 1
            self.bins[new_bin_key] = [row]
            self.centroids[new_bin_key] = self._calculate_centroid(self.bins[new_bin_key])
            if self.verbose:
                print(f"Created a new bin: {new_bin_key}")

    def _find_closest_bin(self, row):
        min_distance = float('inf')
        assigned_bin = None
        for bin_key, centroid in self.centroids.items():
            distance = jensenshannon(row, centroid)
            if np.isfinite(distance) and distance < self.threshold and distance < min_distance:
                assigned_bin = bin_key
                min_distance = distance
        return assigned_bin, min_distance

    def _assign_labels(self, X):
        labels = np.full(len(X), -1)
        for i in range(len(X)):
            assigned_bin, _ = self._find_closest_bin(X.iloc[i].tolist())
            if assigned_bin is not None:
                labels[i] = assigned_bin

        self._handle_small_bins_and_remap(labels)
        self._remap_labels(labels)  # New method for remapping
        return labels

    def _remap_labels(self, labels):
        unique_labels = np.unique(labels)
        mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
        
        # Special case for outlier bin
        if -1 in mapping:
            mapping[-1] = -1  # Make sure outlier bin remains as -1
        
        # Apply remapping
        for i in range(len(labels)):
            labels[i] = mapping[labels[i]]

        # Update bins and centroids accordingly
        new_bins = {}
        new_centroids = {}
        for old, new in mapping.items():
            if old in self.bins:
                new_bins[new] = self.bins[old]
                # Check if the key exists in self.centroids
                if old in self.centroids:
                    new_centroids[new] = self.centroids[old]
        
        self.bins = new_bins
        self.centroids = new_centroids



    def _handle_small_bins_and_remap(self, labels):
        # Move small bins to the outlier bin and remove them
        for bin_key in list(self.bins.keys()):
            if self.min_bin_size is not None and len(self.bins[bin_key]) < self.min_bin_size:
                if -1 not in self.bins:
                    self.bins[-1] = []
                self.bins[-1].extend(self.bins.pop(bin_key))
                del self.centroids[bin_key]
                if self.verbose:
                    print(f"Removed bin {bin_key} due to insufficient size.")

        # Adjust labels for moved data
        labels[np.isin(labels, list(self.bins.keys()), invert=True)] = -1

        # Remap bin keys to be sequential (excluding outlier bin)
        keys_to_remap = sorted([key for key in self.bins.keys() if key != -1])
        for i, old_key in enumerate(keys_to_remap):
            if old_key != i:
                self.bins[i] = self.bins.pop(old_key)
                self.centroids[i] = self.centroids.pop(old_key)
                labels[labels == old_key] = i
                
                

