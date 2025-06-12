GanblrAdapter
=============

A class to adapt the GANBLR model to the KatabaticModelSPI interface. This class provides an easy interface for loading, fitting, and generating data using the GANBLR model.

Attributes
----------

- **type** (`str`):
  Specifies whether the model type is 'discrete' or 'continuous'. Default is 'discrete'.

- **constraints** (`any`):
  Constraints for the model. Default is None.

- **batch_size** (`int`):
  Batch size for training the model. Default is None.

- **epochs** (`int`):
  Number of epochs for training the model. Default is None.

- **training_sample_size** (`int`):
  Size of the training sample. Initialized to 0.

Methods
--------

**load_model()**
   Initializes and returns an instance of the GANBLR model.

**load_data(data_pathname)**
   Loads data from the specified pathname.

   - **Parameters:**
     - `data_pathname`: Pathname of the data to be loaded.

**fit(X_train, y_train, k=0, epochs=10, batch_size=64)**
   Fits the GANBLR model using the provided training data.

   - **Parameters:**
     - `X_train`: Training features.
     - `y_train`: Training labels.
     - `k`: Number of parent nodes (default: 0).
     - `epochs`: Number of epochs for training (default: 10).
     - `batch_size`: Batch size for training (default: 64).

**generate(size=None)**
   Generates data from the GANBLR model. If `size` is not specified, it defaults to the training sample size.

   - **Parameters:**
     - `size`: Number of data samples to generate. Defaults to None.

   - **Returns:**
     Generated data samples.