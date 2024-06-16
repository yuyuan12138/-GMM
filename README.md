# Gaussian Mixture Model (GMM) Implementation

This Python script provides an implementation of the Expectation-Maximization (EM) algorithm for fitting a Gaussian Mixture Model to a dataset. It is particularly useful for statistical data analysis and clustering, where the underlying distribution of the data is assumed to be a mixture of several Gaussian distributions.

## Description

The script defines two main components:

A function generate_gmm_data to generate synthetic data from a specified Gaussian Mixture Model.
A class GaussianMixtureModel that implements the EM algorithm to fit a GMM to given data, allowing estimation of component weights, means, and covariances.

## Installation

To run this script, you need Python installed on your system along with the numpy library. You can install numpy using pip if it's not already installed:

```bash
pip install numpy
```

## Usage

To use this script, simply clone this repository or copy the script to your local machine. You can run the script directly if you want to use the default parameters defined in the if __name__ == "__main__": block. Modify the parameters as needed to fit your specific data analysis needs.

## Example

Here is a quick example of how to use the script to generate data and fit a model:

```python
import numpy as np

# Parameters
weights = np.array([0.6, 0.4])
means = np.array([[0, 0], [3, 3]])
covariances = np.array([[[1, 0.5], [0.5, 1]], [[1, 0.5], [0.5, 1]]])

# Generate data
data = generate_gmm_data(num_samples=600, weights=weights, means=means, covariances=covariances)

# Create and fit the model
model = GaussianMixtureModel(num_components=2, num_iters=100)
model.fit(data)

# Outputs
print("Estimated Weights:", model.weights)
print("Estimated Means:\n", model.means)
print("Estimated Covariances:\n", model.covariances)
```

## Features

Generate synthetic data from a Gaussian Mixture Model.
Fit a Gaussian Mixture Model to data using the EM algorithm.
Extract component weights, means, and covariances from the fitted model.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your enhancements. For major changes, please open an issue first to discuss what you would like to change.
