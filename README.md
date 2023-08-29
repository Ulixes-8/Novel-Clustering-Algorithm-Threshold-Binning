
# Threshold Binning Clustering Algorithm

## Table of Contents
1. Introduction
2. Installation
3. Usage
4. Parameters
5. Methods
6. Examples
7. Performance Considerations
8. Contributing
9. License

### Introduction
The Threshold Binning Clustering Algorithm is a custom clustering algorithm that works by grouping data points into bins based on a distance threshold. The algorithm uses the Jensen-Shannon distance as a metric to compare the similarity between data points and centroids of existing bins. It's especially useful for high-dimensional data where traditional clustering algorithms may not perform well.

### Installation
To use this algorithm, simply add the ThresholdBinning class to your Python project. Ensure that you also have numpy, pandas, scipy, and scikit-learn installed. You can install these packages via pip if you haven't already:

### Usage
You can use the Threshold Binning Clustering Algorithm as you would use any scikit-learn estimator. Here's a quick example:

from threshold_binning import ThresholdBinning
import pandas as pd
data = pd.read_csv("your_data.csv")
model = ThresholdBinning(max_bins=5, threshold=0.1, min_bin_size=5, verbose=False)
labels = model.fit_predict(data)

### The ThresholdBinning class constructor accepts the following parameters:

* max_bins: (int, default=5) The maximum number of bins to create.
* threshold: (float, default=0.1) The distance threshold for assigning data points to bins.
* min_bin_size: (int or None, default=None) Minimum number of samples required in a bin. If None, no minimum size is enforced.
* verbose: (bool, default=False) Verbosity flag.

### Methods
* fit(X, y=None): Fits the model to the data X.
* fit_predict(X, y=None): Fits the model to the data X and returns the labels of the bins.

### Examples
Examples can be found in the examples/ directory in the repository. This will help you understand how to use the algorithm for different types of data.

### Performance Considerations
The Threshold Binning Clustering Algorithm is computationally intensive for large datasets. Performance optimization is an ongoing effort, and contributions are welcome.

### Authors and Acknowledgements 
The algorithm was primarily authored by Tariq U. Hawili during a data science internship to solve a very specific clustering problem on a proprietary dataset. Special thanks to James Bower for his contribution of the min_bin_size hyperparameter and guidance. Further thanks to Ross Massey for his contribution to the Threshold Binning algorithm. 

### Contributing
If you find any bugs, have feature requests, or would like to contribute to the development of this algorithm, please feel free to open an issue or submit a pull request on the GitHub repository.

### License