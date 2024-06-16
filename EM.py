import numpy as np

def generate_gmm_data(num_samples, weights, means, covariances):

    num_samples_component = (num_samples * weights).astype(int)
    data = np.vstack([
        np.random.multivariate_normal(means[0], covariances[0], num_samples_component[0]),
        np.random.multivariate_normal(means[1], covariances[1], num_samples_component[1])
    ])
    np.random.shuffle(data)
    return data

class GaussianMixtureModel:
    def __init__(self, num_components, num_iters):
        self.num_components = num_components
        self.num_iters = num_iters
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []

    def initialize_parameters(self, data):
        n_samples, n_features = data.shape
        self.weights = np.ones(self.num_components) / self.num_components
        self.means = data[np.random.choice(n_samples, self.num_components, False)]
        self.covariances = np.array([np.eye(n_features)] * self.num_components)

    def gaussian_pdf(self, X, mean, covariance):
        n_features = X.shape[1]
        diff = X - mean
        exponent = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1))
        return exponent / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(covariance))

    def e_step(self, data):
        responsibilities = np.zeros((data.shape[0], self.num_components))
        for i in range(self.num_components):
            responsibilities[:, i] = self.weights[i] * self.gaussian_pdf(data, self.means[i], self.covariances[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, data, responsibilities):
        weighted_sum = responsibilities.sum(axis=0)
        self.weights = weighted_sum / data.shape[0]
        for i in range(self.num_components):
            weighted_data_sum = np.sum(responsibilities[:, i, np.newaxis] * data, axis=0)
            self.means[i] = weighted_data_sum / weighted_sum[i]
            diff = data - self.means[i]
            self.covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / weighted_sum[i]

    def compute_log_likelihood(self, data):
        log_likelihood = 0
        for i in range(self.num_components):
            log_likelihood += self.weights[i] * self.gaussian_pdf(data, self.means[i], self.covariances[i])
        return np.log(log_likelihood).sum()

    def fit(self, data):
        self.initialize_parameters(data)
        for _ in range(self.num_iters):
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            log_likelihood = self.compute_log_likelihood(data)
            self.log_likelihood_history.append(log_likelihood)
            if len(self.log_likelihood_history) > 1 and np.abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < 1e-5:
                break

if __name__ == "__main__":

    np.random.seed(42)

    weights = np.array([0.6, 0.4])
    means = np.array([[0, 0], [3, 3]])  
    covariances = np.array([[[1, 0.5], [0.5, 1]], 
                            [[1, 0.5], [0.5, 1]]])  
    
    data = generate_gmm_data(num_samples=600, weights=weights, means=means, covariances=covariances)

    model = GaussianMixtureModel(num_components=len(means), num_iters=100)
    model.fit(data)

    print("Estimated Weights:", model.weights)
    print("Estimated Means:\n", model.means)
    print("Estimated Covariances:\n", model.covariances)

