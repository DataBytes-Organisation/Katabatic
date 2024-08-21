kDB Algorithm
=============

The kDB algorithm constructs a dependency graph for the Bayesian network. It is implemented in the `kdb.py` file.

Methods
--------

**build_graph(X, y, k=2)**
   Constructs a k-dependency Bayesian network graph.

   - **Parameters:**
     - `X`: Input data (features).
     - `y`: Labels.
     - `k`: Number of parent nodes (default: 2).

   - **Returns:**
     A list of graph edges.

**get_cross_table(*cols, apply_wt=False)**
   Generates a cross table from input columns.

   - **Parameters:**
     - `cols`: Columns for cross table generation.
     - `apply_wt`: Whether to apply weights (default: False).

   - **Returns:**
     A tuple containing:
     - The cross table as a NumPy array.
     - A list of unique values for all columns.
     - A list of unique values for individual columns.

**_get_dependencies_without_y(variables, y_name, kdb_edges)**
   Finds the dependencies of each variable without considering `y`.

   - **Parameters:**
     - `variables`: List of variable names.
     - `y_name`: Class name.
     - `kdb_edges`: List of tuples representing edges (source, target).

   - **Returns:**
     A dictionary of dependencies.

**_add_uniform(X, weight=1.0)**
   Adds a uniform distribution to the data.

   - **Parameters:**
     - `X`: Input data, a NumPy array or pandas DataFrame.
     - `weight`: Weight for the uniform distribution (default: 1.0).

   - **Returns:**
     The modified data with uniform distribution.

**_normalize_by_column(array)**
   Normalizes the array by columns.

   - **Parameters:**
     - `array`: Input array to normalize.

   - **Returns:**
     The normalized array.

**_smoothing(cct, d)**
   Probability smoothing for kDB.

   - **Parameters:**
     - `cct`: Cross count table with shape (x0, *parents).
     - `d`: Dimension of `cct`.

   - **Returns:**
     A smoothed joint probability table.

**get_high_order_feature(X, col, evidence_cols, feature_uniques)**
   Encodes the high-order feature of `X[col]` given evidence from `X[evidence_cols`.

   - **Parameters:**
     - `X`: Input data.
     - `col`: Column to encode.
     - `evidence_cols`: List of evidence columns.
     - `feature_uniques`: Unique values for features.

   - **Returns:**
     An encoded high-order feature.

**get_high_order_constraints(X, col, evidence_cols, feature_uniques)**
   Gets high-order constraints for the feature.

   - **Parameters:**
     - `X`: Input data.
     - `col`: Column to encode.
     - `evidence_cols`: List of evidence columns.
     - `feature_uniques`: Unique values for features.

   - **Returns:**
     High-order constraints.

Classes
--------

**KdbHighOrderFeatureEncoder**
   Encodes high-order features for the kDB model.

   - **Class Properties:**
     - `feature_uniques_`: Unique values for features.
     - `dependencies_`: Dependencies for features.
     - `ohe_`: OneHotEncoder instance.

   - **Class Methods:**
     - `fit(X, y, k=0)`: Fits the encoder to the data.
     - `transform(X, return_constraints=False, use_ohe=True)`: Transforms the input data.
     - `fit_transform(X, y, k=0, return_constraints=False)`: Fits the encoder and transforms the data.
