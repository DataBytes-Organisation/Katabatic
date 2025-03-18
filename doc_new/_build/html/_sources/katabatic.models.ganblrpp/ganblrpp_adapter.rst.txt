GanblrppAdapter Class
=====================

The `GanblrppAdapter` class adapts the `GANBLRPP` model to the Katabatic Model SPI interface. It facilitates model initialization, data loading, model fitting, and data generation, bridging the gap between Katabatic's requirements and the GANBLR++ model's functionality.


Class Properties
----------------

- **type** (`str`):
  Type of model to use ("discrete" by default).

- **model**:
  Instance of `GANBLRPP` for model operations.

- **constraints**:
  Constraints for the model (not used in this class).

- **batch_size** (`int`):
  Size of batches for training.

- **epochs** (`int`):
  Number of epochs for training.

- **training_sample_size** (`int`):
  Size of the training sample.

- **numerical_columns** (`list`):
  List of indices for numerical columns in the dataset.

- **random_state**:
  Seed for random number generation.

Class Methods
--------------

**__init__(model_type="discrete", numerical_columns=None, random_state=None)**
   Initializes the `GanblrppAdapter` with model type, numerical columns, and optional random state.

   - **Parameters:**
     - `model_type`: str, default="discrete". Type of model to use.
     - `numerical_columns`: list, optional. List of indices for numerical columns.
     - `random_state`: int, optional. Seed for random number generation.

**load_model() -> GANBLRPP**
   Initializes and loads the `GANBLRPP` model.

   - **Returns:**
     The initialized `GANBLRPP` model.

   - **Raises:**
     - `ValueError`: If `numerical_columns` is not provided.
     - `RuntimeError`: If the model initialization fails.

**load_data(data_pathname) -> pd.DataFrame**
   Loads data from a CSV file.

   - **Parameters:**
     - `data_pathname`: str. Path to the CSV file containing the data.

   - **Returns:**
     Pandas DataFrame containing the loaded data.

   - **Raises:**
     - `Exception`: If there is an error loading the data.

**fit(X_train, y_train, k=0, epochs=10, batch_size=64)**
   Fits the `GANBLRPP` model to the training data.

   - **Parameters:**
     - `X_train`: pd.DataFrame or np.ndarray. Training data features.
     - `y_train`: pd.Series or np.ndarray. Training data labels.
     - `k`: int, default=0. Parameter k for the GANBLRPP model.
     - `epochs`: int, default=10. Number of training epochs.
     - `batch_size`: int, default=64. Size of batches for training.

   - **Returns:**
     None

   - **Raises:**
     - `RuntimeError`: If the model is not initialized.
     - `Exception`: If there is an error during model training.

**generate(size=None) -> pd.DataFrame**
   Generates synthetic data using the `GANBLRPP` model.

   - **Parameters:**
     - `size`: int or None. Size of the synthetic data to generate. Defaults to the size of the training data.

   - **Returns:**
     Pandas DataFrame containing the generated data.

   - **Raises:**
     - `RuntimeError`: If the model is not initialized.
     - `Exception`: If there is an error during data generation.
