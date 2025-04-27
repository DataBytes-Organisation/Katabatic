Utility Functions
=================

The utility functions are implemented in the `utils.py` file and provide various support functions for data preparation and model constraints.

Classes
--------

**softmax_weight**
   Constrains weight tensors to be under softmax.

**DataUtils**
   Provides data utilities for preparation before training.

Functions
---------

**elr_loss(KL_LOSS)**
   Defines a custom loss function.

   - **Parameters:**
     - `KL_LOSS`: The KL loss value.

   - **Returns:**
     A custom loss function.

**KL_loss(prob_fake)**
   Calculates the KL loss.

   - **Parameters:**
     - `prob_fake`: Probability of the fake data.

   - **Returns:**
     The KL loss value.

**get_lr(input_dim, output_dim, constraint=None, KL_LOSS=0)**
   Creates a logistic regression model.

   - **Parameters:**
     - `input_dim`: Dimension of the input features.
     - `output_dim`: Dimension of the output.
     - `constraint`: Optional constraint for the model (default: None).
     - `KL_LOSS`: Optional KL loss value (default: 0).

   - **Returns:**
     A logistic regression model.

**sample(*arrays, n=None, frac=None, random_state=None)**
   Generates random samples from the given arrays.

   - **Parameters:**
     - `arrays`: Arrays to sample from.
     - `n`: Number of samples to generate (default: None).
     - `frac`: Fraction of samples to generate (default: None).
     - `random_state`: Random seed for reproducibility (default: None).

   - **Returns:**
     Random samples from the arrays.

**get_demo_data(name='adult')**
   Downloads a demo dataset from the internet.

   - **Parameters:**
     - `name`: Name of the dataset to download (default: 'adult').

   - **Returns:**
     The downloaded dataset.
