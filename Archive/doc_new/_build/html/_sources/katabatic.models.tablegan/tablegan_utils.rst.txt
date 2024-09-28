Utility Functions
=================

The utility functions are implemented in the `utils.py` file and provide various support functions for data preparation, model constraints, and logistic regression operations.

Classes
-------

- **softmax_weight**
  
  A class that constrains weight tensors to be normalized using the softmax function, ensuring that the values sum up to 1.

- **DataUtils**
  
  A utility class that provides helper functions for preparing data before training machine learning models.

Functions
---------

- **elr_loss(KL_LOSS)**
  
  Defines a custom loss function that integrates the KL loss into the overall loss calculation for model training.
  
  - **Parameters:**
    - `KL_LOSS` (`float`): The Kullback-Leibler (KL) loss value that is integrated into the loss function.
  
  - **Returns:**
    A custom loss function that can be used in training models.

- **KL_loss(prob_fake)**
  
  Calculates the Kullback-Leibler (KL) divergence loss, which measures how one probability distribution diverges from a second, expected distribution.
  
  - **Parameters:**
    - `prob_fake` (`numpy.array`): The probability distribution of the fake (generated) data.
  
  - **Returns:**
    The calculated KL loss value.

- **get_lr(input_dim, output_dim, constraint=None, KL_LOSS=0)**
  
  Creates and returns a logistic regression model, with optional constraints and KL loss integration.
  
  - **Parameters:**
    - `input_dim` (`int`): The dimension of the input features.
    - `output_dim` (`int`): The dimension of the output labels.
    - `constraint` (`callable`, optional): A constraint function applied to the logistic regression model weights (default: None).
    - `KL_LOSS` (`float`, optional): The Kullback-Leibler loss value to be incorporated into the model (default: 0).
  
  - **Returns:**
    A logistic regression model object.

- **sample(*arrays, n=None, frac=None, random_state=None)**
  
  Generates random samples from the provided arrays, either by specifying the number of samples or the fraction of the arrays to sample.
  
  - **Parameters:**
    - `arrays` (`list`): The input arrays to sample from.
    - `n` (`int`, optional): The number of samples to generate (default: None).
    - `frac` (`float`, optional): The fraction of the arrays to sample (default: None).
    - `random_state` (`int`, optional): A random seed to ensure reproducibility (default: None).
  
  - **Returns:**
    Random samples from the provided arrays.

- **get_demo_data(name='adult')**
  
  Downloads and returns a demo dataset from the internet. By default, it downloads the 'adult' dataset, but other dataset names can be specified.
  
  - **Parameters:**
    - `name` (`str`, optional): The name of the dataset to download (default: 'adult').
  
  - **Returns:**
    The downloaded dataset as a Pandas DataFrame.
