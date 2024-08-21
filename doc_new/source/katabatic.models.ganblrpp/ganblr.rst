GANBLR Model Class
===================
The GANBLR model combines a Bayesian network and a neural network, leveraging the k-Dependency Bayesian (kDB) algorithm for building a dependency graph and using various utility functions for data preparation and model constraints.

===================
Defined in `ganblr.py`

Class Properties
----------------

- **_d**:
  Placeholder for data.

- **__gen_weights**:
  Placeholder for generator weights.

- **batch_size** (`int`):
  Batch size for training.

- **epochs** (`int`):
  Number of epochs for training.

- **k**:
  Parameter for the model.

- **constraints**:
  Constraints for the model.

- **_ordinal_encoder**:
  Ordinal encoder for preprocessing data.

- **_label_encoder**:
  Label encoder for preprocessing data.

Class Methods
--------------

**__init__()**
   Initializes the model with default values.

**fit(x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=0)**
   Fits the model to the given data.

   - **Parameters:**
     - `x`: Input data.
     - `y`: Labels for the data.
     - `k`: Parameter for the model (default: 0).
     - `batch_size`: Size of batches for training (default: 32).
     - `epochs`: Number of training epochs (default: 10).
     - `warmup_epochs`: Number of warmup epochs (default: 1).
     - `verbose`: Verbosity level (default: 0).

   - **Returns:**
     The fitted model.

**_augment_cpd(d, size=None, verbose=0)**
   Augments the Conditional Probability Distribution (CPD).

   - **Parameters:**
     - `d`: Data.
     - `size`: Size of the sample (default: None).
     - `verbose`: Verbosity level (default: 0).

   - **Returns:**
     The augmented data.

**_warmup_run(epochs, verbose=None)**
   Runs a warmup phase.

   - **Parameters:**
     - `epochs`: Number of epochs for warmup.
     - `verbose`: Verbosity level (default: None).

   - **Returns:**
     The history of the warmup run.

**_run_generator(loss)**
   Runs the generator model.

   - **Parameters:**
     - `loss`: Loss function.

   - **Returns:**
     The history of the generator run.

**_discrim()**
   Creates a discriminator model.

   - **Returns:**
     The discriminator model.

**_sample(size=None, verbose=0)**
   Generates synthetic data in ordinal encoding format.

   - **Parameters:**
     - `size`: Size of the sample (default: None).
     - `verbose`: Verbosity level (default: 0).

   - **Returns:**
     The sampled data.
