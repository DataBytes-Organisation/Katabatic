Utility Classes
===============

The utility classes are implemented in the `utility.py` file and provide various support functionalities for model constraints and data preparation.

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

Classes
--------

**softmax_weight**
   Constrains weight tensors to be under softmax.

   - **Defined in:** `softmax_weight.py`

   - **Properties:**
     - **feature_idxs** (`list`): List of tuples indicating the start and end indices for each feature in the weight tensor.

   - **Methods:**
     - **__init__(feature_uniques)**
     Initializes the constraint with unique feature values.
       
       - **Parameters:**
         - `feature_uniques`: `np.ndarray` or list of int. Unique values for each feature used to compute indices for softmax constraint.
       
       - **Returns:**
         None.

     - **__call__(w)**
       Applies the softmax constraint to the weight tensor.
       
       - **Parameters:**
         - `w`: `tf.Tensor`. Weight tensor to which the constraint is applied.
       
       - **Returns:**
         `tf.Tensor`: The constrained weight tensor.

     - **get_config()**
       Returns the configuration of the constraint.
       
       - **Returns:**
         `dict`: Configuration dictionary containing `feature_idxs`.

**DataUtils**
   Provides utility functions for data preparation before training.

   - **Defined in:** `data_utils.py`

   - **Properties:**
     - **x** (`np.ndarray`): The feature data used for training.
     - **y** (`np.ndarray`): The target labels associated with the feature data.
     - **data_size** (`int`): Number of samples in the dataset.
     - **num_features** (`int`): Number of features in the dataset.
     - **num_classes** (`int`): Number of unique classes in the target labels.
     - **class_counts** (`np.ndarray`): Counts of each class in the target labels.
     - **feature_uniques** (`list`): List of unique values for each feature.
     - **constraint_positions** (`np.ndarray` or `None`): Positions of constraints for high-order features.
     - **_kdbe** (`KdbHighOrderFeatureEncoder` or `None`): Instance of the `KdbHighOrderFeatureEncoder` for feature encoding.
     - **__kdbe_x** (`np.ndarray` or `None`): Transformed feature data after applying kDB encoding.

   - **Methods:**
     - **__init__(x, y)**
     Initializes the `DataUtils` with the provided feature data and target labels.
       
       - **Parameters:**
         - `x`: `np.ndarray`. Feature data.
         - `y`: `np.ndarray`. Target labels.
       
       - **Returns:**
         None.

     - **get_categories(idxs=None)**
       Retrieves categories for encoded features.
       
       - **Parameters:**
         - `idxs`: list of int or `None`. Indices of features to retrieve categories for. If `None`, retrieves categories for all features.
       
       - **Returns:**
         `list`: List of categories for the specified features.

     - **get_kdbe_x(k=0, dense_format=True)**
       Transforms feature data into high-order features using kDB encoding.
       
       - **Parameters:**
         - `k`: int, default=0. `k` value for the kDB model.
         - `dense_format`: bool, default=True. Whether to return the transformed data in dense format.
       
       - **Returns:**
         `np.ndarray`: Transformed feature data.
         If `dense_format=True`, returns a dense NumPy array.
         Also updates `constraint_positions` with the positions of constraints.

     - **clear()**
       Clears the kDB encoder and transformed data.
       
       - **Returns:**
         None.
