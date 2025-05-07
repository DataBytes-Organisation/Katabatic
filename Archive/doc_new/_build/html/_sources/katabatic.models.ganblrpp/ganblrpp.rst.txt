DMMDiscritizer Class
====================

The `DMMDiscritizer` class performs discretization using a mixture model approach. It scales the data, applies Bayesian Gaussian Mixture Models, and transforms the data into discrete values. It also includes methods for inverse transformation of discretized data back to its original form.

Class Properties
----------------

- **__dmm_params**:
  Parameters for the Bayesian Gaussian Mixture Model.

- **__scaler**:
  Min-Max scaler for data normalization.

- **__dmms**:
  List to store Bayesian Gaussian Mixture Models for each feature.

- **__arr_mu**:
  List to store means of the Gaussian distributions.

- **__arr_sigma**:
  List to store standard deviations of the Gaussian distributions.

- **_random_state**:
  Random seed for reproducibility.

Class Methods
--------------

**__init__(random_state)**
   Initializes the DMMDiscritizer with parameters and scaler.

   - **Parameters:**
     - `random_state`: Seed for random number generation.

**fit(x)**
   Fits the discretizer to the provided data.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Numeric data to be discretized.

   - **Returns:**
     The fitted `DMMDiscritizer` instance.

**transform(x) -> np.ndarray**
   Transforms the data using the fitted discretizer.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Numeric data to be transformed.

   - **Returns:**
     2D numpy array of shape (n_samples, n_features). Discretized data.

**fit_transform(x) -> np.ndarray**
   Fits the discretizer and transforms the data.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Numeric data to be discretized and transformed.

   - **Returns:**
     2D numpy array of shape (n_samples, n_features). Discretized data.

**inverse_transform(x, verbose=1) -> np.ndarray**
   Converts discretized data back to its original continuous form.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Discretized data.
     - `verbose`: int, default=1. Controls verbosity of the operation.

   - **Returns:**
     2D numpy array of shape (n_samples, n_features). Reverted data.

**__sample_from_truncnorm(bins, mu, sigma, random_state=None)**
   Samples data from a truncated normal distribution.

   - **Parameters:**
     - `bins`: 1D numpy array of integer bins.
     - `mu`: 1D numpy array of means for the normal distribution.
     - `sigma`: 1D numpy array of standard deviations for the normal distribution.
     - `random_state`: int or None. Seed for random number generation.

   - **Returns:**
     1D numpy array of sampled results.

GANBLRPP Class
================

The `GANBLRPP` class implements the GANBLR++ model, which combines generative adversarial networks with discretization techniques. It uses the `DMMDiscritizer` for data preprocessing and the `GANBLR` model for generating synthetic data.

Class Properties
----------------

- **__discritizer**:
  Instance of `DMMDiscritizer` for data preprocessing.

- **__ganblr**:
  Instance of `GANBLR` for generative adversarial network functionality.

- **_numerical_columns**:
  List of indices for numerical columns in the dataset.

Class Methods
--------------

**__init__(numerical_columns, random_state=None)**
   Initializes the GANBLR++ model with numerical column indices and optional random state.

   - **Parameters:**
     - `numerical_columns`: List of indices for numerical columns.
     - `random_state`: int, RandomState instance or None. Seed for random number generation.

**fit(x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=1)**
   Fits the GANBLR++ model to the provided data.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Dataset to fit the model.
     - `y`: 1D numpy array of shape (n_samples,). Labels for the dataset.
     - `k`: int, default=0. Parameter k for the GANBLR model.
     - `batch_size`: int, default=32. Size of batches for training.
     - `epochs`: int, default=10. Number of training epochs.
     - `warmup_epochs`: int, default=1. Number of warmup epochs.
     - `verbose`: int, default=1. Controls verbosity of the training process.

   - **Returns:**
     The fitted `GANBLRPP` instance.

**sample(size=None, verbose=1)**
   Generates synthetic data using the GANBLR++ model.

   - **Parameters:**
     - `size`: int or None. Size of the synthetic data to generate.
     - `verbose`: int, default=1. Controls verbosity of the sampling process.

   - **Returns:**
     2D numpy array of synthetic data.

**evaluate(x, y, model='lr')**
   Evaluates the model using a TSTR (Training on Synthetic data, Testing on Real data) approach.

   - **Parameters:**
     - `x`: 2D numpy array of shape (n_samples, n_features). Test dataset.
     - `y`: 1D numpy array of shape (n_samples,). Labels for the test dataset.
     - `model`: str or object. Model to use for evaluation. Options are 'lr', 'mlp', 'rf', or a custom model with `fit` and `predict` methods.

   - **Returns:**
     float. Accuracy score of the evaluation.
