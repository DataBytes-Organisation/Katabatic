TableGANAdapter
===============

The `TableGANAdapter` class serves as an adapter for interfacing with the `TableGAN` model. It extends the `KatabaticModelSPI` and allows for loading, fitting, and generating synthetic data using the TableGAN model. This adapter includes functionality to handle privacy settings, data preprocessing, and model training.

Class Structure
---------------

.. autoclass:: TableGANAdapter
    :members:
    :show-inheritance:

Attributes
----------

- **type** (`str`): Defines the type of data handled by the adapter (default: `'continuous'`).
- **privacy_setting** (`str`): Sets the privacy level of the model ('low', 'medium', or 'high').
- **constraints** (`NoneType`): Currently not in use but reserved for future constraints settings.
- **batch_size** (`int`): Defines the batch size for training (default: `64`).
- **epochs** (`int`): Number of training epochs (default: `100`).
- **model** (`TableGAN`): Instance of the TableGAN model.
- **scaler** (`StandardScaler`): Scaler used for preprocessing continuous data.
- **label_encoder** (`LabelEncoder`): Encoder used for processing categorical labels.
- **input_dim** (`int`): Input dimensionality of the data.
- **label_dim** (`int`): Label dimensionality of the data.
- **training_sample_size** (`int`): Number of samples used during training.

Methods
-------

- **__init__(self, type='continuous', privacy_setting='low')**
  
  Initializes the `TableGANAdapter` class, setting parameters such as data type, privacy level, and default model parameters.

- **load_model(self)**
  
  Loads and initializes the `TableGAN` model based on the input and label dimensions. Adjusts privacy parameters (`delta_mean`, `delta_sd`) according to the specified `privacy_setting`.

- **load_data(self, data_pathname)**
  
  Loads training data from the specified `data_pathname`. Handles CSV files and returns the data as a Pandas DataFrame.

- **fit(self, X_train, y_train, epochs=None, batch_size=None)**
  
  Trains the `TableGAN` model on the provided training data (`X_train`, `y_train`). Preprocesses the data, sets input and label dimensions, and optionally overrides the number of epochs and batch size.

- **generate(self, size=None)**
  
  Generates synthetic data using the trained `TableGAN` model. If the model is not trained, raises an error. The size of generated data defaults to the training sample size unless otherwise specified.

Usage Example
-------------

Below is a usage example that shows how to initialize and use the `TableGANAdapter` class for training and generating synthetic data:

.. code-block:: python

    from katabatic.models import TableGANAdapter

    # Initialize the adapter with medium privacy
    adapter = TableGANAdapter(type='continuous', privacy_setting='medium')

    # Load data
    data = adapter.load_data('training_data.csv')

    # Preprocess and fit the model
    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]
    adapter.fit(X_train, y_train)

    # Generate synthetic data
    synthetic_data = adapter.generate(size=1000)

