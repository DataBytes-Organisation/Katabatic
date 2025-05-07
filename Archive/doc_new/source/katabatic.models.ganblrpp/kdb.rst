KdbHighOrderFeatureEncoder Class
================================

The `KdbHighOrderFeatureEncoder` class encodes high-order features using the kDB algorithm. It retrieves dependencies between features, transforms data into high-order features, and optionally returns constraints information.

Defined in `kdb.py`

Class Properties
----------------

- **dependencies_** (`dict`):
  A dictionary storing the dependencies between features.

- **constraints_** (`np.ndarray`):
  Constraints information for high-order features.

- **have_value_idxs_** (`list`):
  Indices indicating the presence of values for constraints.

- **feature_uniques_** (`list`):
  List of unique values for each feature.

- **high_order_feature_uniques_** (`list`):
  List of unique values for high-order features.

- **edges_** (`list`):
  List of edges representing the kDB graph.

- **ohe_** (`OneHotEncoder`):
  OneHotEncoder instance for encoding features.

- **k** (`int`):
  Value of `k` for the kDB model.

Methods
-------

**__init__()**
   Initializes the `KdbHighOrderFeatureEncoder` with default properties.

**fit(X, y, k=0)**
   Fits the `KdbHighOrderFeatureEncoder` to the data and labels.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Data to fit in the encoder.
     - `y`: array_like of shape (n_samples,). Labels to fit in the encoder.
     - `k`: int, default=0. `k` value for the kDB model. `k=0` leads to a OneHotEncoder.

   - **Returns:**
     - `self`: The fitted encoder.

**transform(X, return_constraints=False, use_ohe=True)**
   Transforms data to high-order features.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Data to transform.
     - `return_constraints`: bool, default=False. Whether to return constraint information.
     - `use_ohe`: bool, default=True. Whether to apply one-hot encoding.

   - **Returns:**
     - `X_out`: ndarray of shape (n_samples, n_encoded_features). Transformed input.
     - If `return_constraints=True`, also returns:
     - `constraints`: np.ndarray of constraints.
     - `have_value_idxs`: List of boolean arrays indicating presence of values.

**fit_transform(X, y, k=0, return_constraints=False)**
   Fits the encoder and then transforms the data.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Data to fit and transform.
     - `y`: array_like of shape (n_samples,). Labels to fit and transform.
     - `k`: int, default=0. `k` value for the kDB model.
     - `return_constraints`: bool, default=False. Whether to return constraint information.

   - **Returns:**
     - `X_out`: ndarray of shape (n_samples, n_encoded_features). Transformed input.
     - If `return_constraints=True`, also returns:
     - `constraints`: np.ndarray of constraints.
     - `have_value_idxs`: List of boolean arrays indicating presence of values.

Helper Functions
----------------

**build_graph(X, y, k=2)**
   Constructs a kDB graph based on mutual information.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Features.
     - `y`: array_like of shape (n_samples,). Labels.
     - `k`: int, default=2. Maximum number of parent nodes to consider.

   - **Returns:**
     - `edges`: List of tuples representing edges in the graph.

**get_cross_table(*cols, apply_wt=False)**
   Computes a cross-tabulation table for the given columns.

   - **Parameters:**
     - `*cols`: 1D numpy arrays of integers. Columns to cross-tabulate.
     - `apply_wt`: bool, default=False. Whether to apply weights.

   - **Returns:**
     - `uniq_vals_all_cols`: Tuple of 1D numpy arrays of unique values for each column.
     - `xt`: NumPy array of cross-tabulation results.

**_get_dependencies_without_y(variables, y_name, kdb_edges)**
   Retrieves dependencies of each variable excluding the target variable `y`.

   - **Parameters:**
     - `variables`: List of variable names.
     - `y_name`: Name of the target variable.
     - `kdb_edges`: List of tuples representing edges in the kDB graph.

   - **Returns:**
     - `dependencies`: Dictionary of dependencies for each variable.

**_add_uniform(array, noise=1e-5)**
   Adds uniform probability to avoid zero counts in cross-tabulation.

   - **Parameters:**
     - `array`: NumPy array of counts.
     - `noise`: float, default=1e-5. Amount of noise to add.

   - **Returns:**
     - `result`: NumPy array with added uniform probability.

**_normalize_by_column(array)**
   Normalizes an array by columns.

   - **Parameters:**
     - `array`: NumPy array to normalize.

   - **Returns:**
     - `normalized_array`: NumPy array normalized by columns.

**_smoothing(cct, d)**
   Applies smoothing to a cross-count table to handle zero counts.

   - **Parameters:**
     - `cct`: NumPy array of cross-count table.
     - `d`: int. Dimension of the cross-count table.

   - **Returns:**
     - `jpt`: NumPy array of smoothed joint probability table.

**get_high_order_feature(X, col, evidence_cols, feature_uniques)**
   Encodes high-order features given the evidence columns.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Data.
     - `col`: int. Column index of the feature to encode.
     - `evidence_cols`: List of indices for evidence columns.
     - `feature_uniques`: List of unique values for each feature.

   - **Returns:**
     - `high_order_feature`: NumPy array of high-order features.

**get_high_order_constraints(X, col, evidence_cols, feature_uniques)**
   Finds constraints for high-order features.

   - **Parameters:**
     - `X`: array_like of shape (n_samples, n_features). Data.
     - `col`: int. Column index of the feature.
     - `evidence_cols`: List of indices for evidence columns.
     - `feature_uniques`: List of unique values for each feature.

   - **Returns:**
     - `have_value`: NumPy array of boolean values indicating presence of values.
     - `high_order_constraints`: NumPy array of constraint information.
