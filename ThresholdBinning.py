import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.base import BaseEstimator

 

class ThresholdBinning(BaseEstimator):

       

    def __init__(self, max_bins=5, threshold=0.1, min_bin_size=5, verbose=False):

        self.max_bins = max_bins

        self.threshold = threshold

        self.min_bin_size = min_bin_size

        self.verbose = verbose

        self.bins = {}

        self.centroids = {}

        # Rest of the class here

    def normalize_rows(self, df):

        normalized_df = df.apply(lambda row: (row) / (row.sum()), axis=1)

        return normalized_df

 

    def calculate_centroid(self, bin):

        if not bin:

            return None

       

        num_distributions = len(bin)

        distribution_length = len(bin[0])

       

        # Initialize an array to store the sum of probabilities for each index

        sum_probabilities = np.zeros(distribution_length)

       

        # Calculate the sum of probabilities for each index

        for distribution in bin:

            sum_probabilities += distribution

       

        # Calculate the mean by dividing the sum by the number of distributions

        mean = sum_probabilities / num_distributions

       

        # Return the mean as a centroid probability distribution

        centroid = mean.tolist()

        return centroid

 

    def threshold_binning(self, X):

 

        # Normalize each row of the DataFrame

        X = self.normalize_rows(X)

       

        # Initialize the bins and centroids

        first_row = X.iloc[0]

        first_row = first_row.tolist()

        self.bins = {0: [first_row]}

        self.centroids = {0: self.calculate_centroid(self.bins[0])}

       

        if self.verbose:

            print(f"bins: {self.bins}")

            print(f"centroids: {self.centroids}")

       

        # Iterate over each row in the DataFrame

        for i in range(1, len(X)):

            row = X.iloc[i]

            row = row.tolist()

            assigned_bin = None

            min_distance = float('inf')

           

            # Calculate the rounded Jensen-Shannon distance with each bin

            for bin_key, bin_values in self.bins.items():

                centroid = self.centroids[bin_key]

                rounded_row = np.round(row, decimals=8)  # Adjust the decimal precision as needed

                rounded_centroid = np.round(centroid, decimals=8)  # Adjust the decimal precision as needed

                distance = jensenshannon(rounded_row, rounded_centroid)

                   

                # Check if the distance is valid (not NaN) and below the threshold

                if np.isfinite(distance) and distance < self.threshold and distance < min_distance:

                    assigned_bin = bin_key

                    min_distance = distance

           

            # If a bin is assigned, add the row to the bin and update the centroid

            if assigned_bin is not None:

                self.bins[assigned_bin].append(row)

                self.centroids[assigned_bin] = self.calculate_centroid(self.bins[assigned_bin])

            # If no bin is assigned and the maximum number of bins has not been reached, create a new bin and assign the row to it

            elif len(self.bins) < self.max_bins:

                new_bin_key = max(self.bins.keys()) + 1

                self.bins[new_bin_key] = [row]

                self.centroids[new_bin_key] = self.calculate_centroid(self.bins[new_bin_key])


    def fit_predict(self, X):

       

        self.threshold_binning(X)

 

        # Initial label assignment

        labels = np.full(len(X), -1)

        for i in range(len(X)):

            row = X.iloc[i]

            row = row.tolist()

            assigned_bin = None

            min_distance = float('inf')

 

            # Calculate the rounded Jensen-Shannon distance with each bin

            for bin_key, bin_values in self.bins.items():

                centroid = self.centroids[bin_key]

                rounded_row = np.round(row, decimals=8)  # Adjust the decimal precision as needed

                rounded_centroid = np.round(centroid, decimals=8)  # Adjust the decimal precision as needed

                distance = jensenshannon(rounded_row, rounded_centroid)

 

                # Check if the distance is valid (not NaN) and below the threshold

                if np.isfinite(distance) and distance < self.threshold and distance < min_distance:

                    assigned_bin = bin_key

                    min_distance = distance

                   

#                         # Calculate the rounded Jensen-Shannon distance with each bin

#             for bin_key, bin_values in self.bins.items():

#                 centroid = self.centroids[bin_key]

#                 rounded_row = np.round(row, decimals=8)  # Adjust the decimal precision as needed

#                 rounded_centroid = np.round(centroid, decimals=8)  # Adjust the decimal precision as needed

 

#                 epsilon = 1e-7  # small constant to add to distributions

#                 rounded_row_eps = rounded_row + epsilon

#                 rounded_centroid_eps = rounded_centroid + epsilon

 

#                 distance = jensenshannon(rounded_row_eps, rounded_centroid_eps)

 

#                 # Check if the distance is valid (not NaN) and below the threshold

#                 if np.isfinite(distance) and distance < self.threshold and distance < min_distance:

#                     assigned_bin = bin_key

#                     min_distance = distance

 

 

            # If a bin is assigned, assign the corresponding label

            if assigned_bin is not None:

                labels[i] = assigned_bin

 

        # Move small bins to the outlier bin and remove them

        for bin_key in list(self.bins.keys()):

            if len(self.bins[bin_key]) < self.min_bin_size:

                if -1 not in self.bins:

                    self.bins[-1] = []

                self.bins[-1].extend(self.bins[bin_key])

                del self.bins[bin_key]

                del self.centroids[bin_key]

 

        # Adjust labels for moved data

        labels[np.isin(labels, list(self.bins.keys()), invert=True)] = -1

 

        # Remap bin keys to be sequential (excluding outlier bin)

        keys_to_remap = [key for key in self.bins.keys() if key != -1]

        keys_to_remap.sort()

        for i, old_key in enumerate(keys_to_remap):

            if old_key != i:

                self.bins[i] = self.bins.pop(old_key)

                self.centroids[i] = self.centroids.pop(old_key)

                labels[labels == old_key] = i

 

        return labels

 