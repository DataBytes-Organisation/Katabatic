```markdown
# CTGAN Pipeline: Errors and Fixes

This document outlines the major errors encountered across the three scripts—**`benchmark_ctgan`**, **`ctgan_adapter`**, and **`run_ctgan`**—along with the corresponding fixes implemented. The errors are organized by script for clarity and better understanding.

The implementation of CTGAN in this project is based on the groundbreaking work by Xu et al. (2019), adapting their methodologies and insights to the Katabatic framework.

---

## `ctgan_adapter.py`

### 1. **Import and Module Path Errors**
- **Error:**  
  Unable to import custom modules from the `katabatic` package, resulting in `ModuleNotFoundError`.

- **Fix:**  
  Added the project root directory to Python's system path to enable importing custom modules.
  ```python
  project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
  sys.path.insert(0, project_root)
  ```

### 2. **Handling Mixed Data Types**
- **Error:**  
  Inconsistent encoding of categorical features leading to data mismatches during transformation and inverse transformation.

- **Fix:**  
  Implemented the `DataTransformer` class to handle both categorical and numerical features systematically.
  ```python
  class DataTransformer:
      def __init__(self, max_clusters=10):
          # Initialization code
          self.encoders = {}
          self.scalers = {}
          # ...
      
      def fit(self, data):
          for column in data.columns:
              if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                  encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                  encoder.fit(data[[column]])
                  self.encoders[column] = encoder
              else:
                  scaler = MinMaxScaler()
                  scaler.fit(data[[column]])
                  self.scalers[column] = scaler
                  # ...
      
      def transform(self, data):
          # Transformation logic
          pass
      
      def inverse_transform(self, data):
          # Inverse transformation logic
          pass
  ```

### 3. **Missing Values in Synthetic Target Variable**
- **Error:**  
  Synthetic data generation resulted in `NaN` values in the target variable, causing errors during evaluation.

- **Fix:**  
  Imputed missing values in the synthetic target variable with the mode to maintain data integrity.
  ```python
  recovered = scaler.inverse_transform(recovered.reshape(-1, 1)).flatten()
  df[column] = recovered
  # Later during evaluation
  y_synthetic = y_synthetic.fillna(y_synthetic.mode().iloc[0])
  ```

### 4. **Dimension Mismatch Between Generator and Discriminator**
- **Error:**  
  Mismatched input dimensions when concatenating noise vectors with conditional vectors, leading to runtime errors.

- **Fix:**  
  Verified and correctly calculated `noise_dim` and `cond_dim`, ensuring proper concatenation.
  ```python
  self.input_dim = noise_dim + cond_dim
  ```

### 5. **Residual Connections Causing Shape Mismatches**
- **Error:**  
  Residual connections in the generator added tensors of different shapes, causing runtime errors during forward passes.

- **Fix:**  
  Ensured matching dimensions in the `ResidualBlock` by adjusting layers or adding transformations.
  ```python
  class ResidualBlock(nn.Module):
      def __init__(self, input_dim, output_dim):
          super(ResidualBlock, self).__init__()
          self.fc = nn.Linear(input_dim, output_dim)
          self.bn = nn.BatchNorm1d(output_dim)
      
      def forward(self, x):
          out = F.leaky_relu(self.bn(self.fc(x)), 0.2)
          if x.shape != out.shape:
              x = F.leaky_relu(self.bn(nn.Linear(x.shape[1], out.shape[1])(x)), 0.2)
          return x + out
  ```

### 6. **Gradient Leakage from Generator to Discriminator**
- **Error:**  
  Gradients from the generator were inadvertently propagating to the discriminator, causing unstable training dynamics.

- **Fix:**  
  Detached fake data from the computation graph when training the discriminator to prevent gradient flow.
  ```python
  fake_cat = torch.cat([fake, c1], dim=1).detach()
  ```

### 7. **Device Mismatch Errors (CPU vs GPU)**
- **Error:**  
  Tensors were located on different devices (CPU vs GPU), causing runtime errors during operations.

- **Fix:**  
  Ensured that all tensors and models were consistently moved to the same device.
  ```python
  self.device = torch.device("cuda:0" if kwargs.get('cuda', True) and torch.cuda.is_available() else "cpu")
  self.generator.to(self.device)
  self.discriminator.to(self.device)
  
  real = data_batch[0].to(self.device)
  noise = torch.randn(current_batch_size, self.embedding_dim, device=self.device)
  c1 = torch.from_numpy(condvec).to(self.device) if condvec is not None else torch.zeros(current_batch_size, self.cond_dim, device=self.device)
  ```

### 8. **Missing or Incorrect Configuration Parameters**
- **Error:**  
  Missing hyperparameters in `config.json` led to unexpected behaviors or crashes.

- **Fix:**  
  Provided default values using `kwargs.get()` and validated the presence of essential configuration keys.
  ```python
  def load_config(config_path="katabatic/models/ctgan/config.json"):
      if not os.path.exists(config_path):
          logging.error(f"Configuration file not found at {config_path}")
          sys.exit(1)
      with open(config_path, "r") as f:
          config = json.load(f)
      required_keys = ["ctgan_params", "evaluation", "visualization"]
      for key in required_keys:
          if key not in config:
              raise KeyError(f"Missing required configuration key: {key}")
      return config
  ```

### 9. **Incorrect Calculation of Evaluation Metrics**
- **Error:**  
  Metrics were inaccurately calculated, leading to misleading evaluations of synthetic data quality.

- **Fix:**  
  Validated metric implementations and handled edge cases appropriately.
  ```python
  "JSD_mean": np.mean(js_divergences) if js_divergences else None,
  "Wasserstein_mean": np.mean(wasserstein_distances) if wasserstein_distances else None
  ```

### 10. **Incorrect Sampling of Conditional Vectors**
- **Error:**  
  Conditional vectors were sampled incorrectly, reducing data diversity and realism.

- **Fix:**  
  Ensured correct one-hot encoding and balanced probability distributions during sampling.
  ```python
  one_hot = encoder.transform([[category]]).flatten()
  cond[i, self.get_condvec_indices(idx)] = one_hot
  ```

### 11. **Batch Size Larger Than Dataset Size**
- **Error:**  
  Set batch size larger than the dataset size, causing empty batches or processing errors.

- **Fix:**  
  Dynamically adjusted batch size based on dataset size.
  ```python
  self.batch_size = min(self.max_batch_size, len(data))
  if self.batch_size < self.max_batch_size:
      logging.info(f"Adjusted batch size to {self.batch_size} due to small dataset size")
  ```

### 12. **Incorrect Reconstruction of Continuous Features**
- **Error:**  
  Continuous features were not accurately inverse transformed, resulting in unrealistic values.

- **Fix:**  
  Handled division by zero, applied clipping, and correctly scaled values back using inverse transformation.
  ```python
  selected_stds[selected_stds == 0] = 1e-6  # Avoid division by zero
  recovered = scaler.inverse_transform(recovered.reshape(-1, 1)).flatten()
  ```

### 13. **Handling Datasets with No Categorical Features**
- **Error:**  
  Scripts failed when processing datasets that lacked categorical features, leading to `None` values or improper handling.

- **Fix:**  
  Added conditional checks to handle absence of categorical columns gracefully.
  ```python
  if not self.discrete_columns:
      return None, None
  ```

### 14. **High Cardinality Categorical Features Causing Memory Issues**
- **Error:**  
  Excessive one-hot encoding led to high-dimensional data and memory overflow.

- **Fix:**  
  Limited the number of Gaussian Mixture Model (GMM) components and filtered inactive components.
  ```python
  self.continuous_gmms[column] = vgm
  component_one_hot = component_one_hot[:, active_components]
  ```

### 15. **Learning Rate and Optimization Issues**
- **Error:**  
  Poor model convergence due to inappropriate learning rates and optimizer settings.

- **Fix:**  
  Tuned learning rates, implemented learning rate schedulers, and set appropriate optimizer betas.
  ```python
  self.schedulerG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=100, gamma=0.5)
  self.schedulerD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=100, gamma=0.5)
  self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.9))
  self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.9))
  ```

### 16. **Gradient Penalty Calculation Errors for WGAN-GP**
- **Error:**  
  Incorrect gradient penalty calculation led to unstable training.

- **Fix:**  
  Correctly computed gradients and integrated gradient penalty into discriminator loss.
  ```python
  gradients = torch.autograd.grad(
      outputs=validity_interpolates,
      inputs=interpolates,
      grad_outputs=torch.ones(validity_interpolates.size(), device=device),
      create_graph=True,
      retain_graph=True,
      only_inputs=True
  )[0]
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
  loss_d = loss_adv + gp + class_loss if self.num_classes > 0 else loss_adv + gp
  ```

### 17. **Incorrect Identification of Target Variable**
- **Error:**  
  Assumed the target variable was always named 'Category', causing misalignment during data processing.

- **Fix:**  
  Dynamically identified the target variable based on dataset structure.
  ```python
  if self.discrete_columns:
      self.target_column = self.discrete_columns[-1]
      self.num_classes = self.transformer.encoders[self.target_column].categories_[0].shape[0]
  else:
      self.target_column = None
      self.num_classes = 0
  ```

### 18. **Handling High-Dimensional Output from DataTransformer**
- **Error:**  
  High-dimensional transformed data caused memory constraints and performance issues.

- **Fix:**  
  Limited the number of GMM components and applied dimensionality reduction techniques where necessary.
  ```python
  self.max_clusters = max_clusters
  active_components = vgm.weights_ > 1e-3
  component_one_hot = component_one_hot[:, active_components]
  ```

### 19. **Saving Synthetic Data When Generation Fails**
- **Error:**  
  Attempted to save synthetic data even when generation failed, leading to corrupted or incomplete files.

- **Fix:**  
  Implemented checks to save only when synthetic data was successfully generated.
  ```python
  if "synthetic_data" in result and result["synthetic_data"] is not None:
      result["synthetic_data"].to_csv(output_file, index=False)
  else:
      logging.warning(f"Failed to generate synthetic data for {result['dataset']}")
  ```

### 20. **Applying Classification Metrics to Numerical Targets**
- **Error:**  
  Used classification metrics (Accuracy, F1 Score) for regression tasks, leading to inappropriate evaluations.

- **Fix:**  
  Separated evaluation logic based on the target variable type, applying appropriate metrics.
  ```python
  if y_real.dtype == 'object' or y_real.dtype.name == 'category':
      # Calculate Accuracy and F1 Score
      accuracy = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average='weighted')
  else:
      # Calculate R-squared
      r2 = r2_score(y_test, y_pred)
  ```

### 21. **Handling One-Hot Encoding Sparse Matrices**
- **Error:**  
  One-hot encoded data as sparse matrices caused concatenation issues with numerical features.

- **Fix:**  
  Set `sparse=False` in `OneHotEncoder` or converted sparse matrices to dense arrays.
  ```python
  encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
  transformed = encoder.transform(data[[column]]).toarray()
  ```

### 22. **Synthetic Target Variable Not Properly Conditioned**
- **Error:**  
  Synthetic target did not accurately reflect conditional distributions, leading to mismatches.

- **Fix:**  
  Ensured that conditional vectors included the target variable correctly and handled encoding/decoding appropriately.
  ```python
  c1 = torch.from_numpy(condvec).to(self.device) if condvec is not None else torch.zeros(current_batch_size, self.cond_dim, device=self.device)
  ```

### 23. **Configuration File Path Errors**
- **Error:**  
  Configuration file (`config.json`) not found due to incorrect path specification, resulting in `FileNotFoundError`.

- **Fix:**  
  Verified and corrected the configuration file path and added error handling to provide informative messages.
  ```python
  def load_config(config_path="katabatic/models/ctgan/config.json"):
      if not os.path.exists(config_path):
          logging.error(f"Configuration file not found at {config_path}")
          sys.exit(1)
      with open(config_path, "r") as f:
          config = json.load(f)
      return config
  ```

---

## `run_ctgan.py`

### 1. **Import and Module Path Errors**
- **Error:**  
  Unable to import custom modules (`CtganAdapter`, `evaluate_ctgan`, `print_evaluation_results`) from the `katabatic` package.

- **Fix:**  
  Added the project root directory to Python's system path.
  ```python
  project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
  sys.path.insert(0, project_root)
  ```

### 2. **Handling Multiple Datasets with Mixed Data Types**
- **Error:**  
  Different datasets had varying feature types and structures, causing preprocessing inconsistencies.

- **Fix:**  
  Systematically converted each dataset into a `pandas.DataFrame` and introduced a synthetic categorical feature.
  ```python
  def load_data():
      datasets = {
          "iris": load_iris(),
          "breast_cancer": load_breast_cancer(),
          "wine": load_wine(),
          "digits": load_digits()
      }
      
      processed_datasets = {}
      
      for name, dataset in datasets.items():
          X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
          y = pd.Series(dataset.target, name="Category")
          X["categorical_feature"] = pd.cut(X.iloc[:, 0], bins=3, labels=["low", "medium", "high"])
          data = pd.concat([X, y], axis=1)
          processed_datasets[name] = data
      
      return processed_datasets
  ```

### 3. **Batch Size Larger Than Dataset Size**
- **Error:**  
  Set batch size larger than the dataset size, causing empty batches or processing errors.

- **Fix:**  
  Dynamically adjusted batch size based on dataset size.
  ```python
  self.batch_size = min(self.max_batch_size, len(data))
  if self.batch_size < self.max_batch_size:
      logging.info(f"Adjusted batch size to {self.batch_size} due to small dataset size")
  ```

### 4. **Saving Synthetic Data When Generation Fails**
- **Error:**  
  Attempted to save synthetic data even when generation failed, leading to corrupted or incomplete files.

- **Fix:**  
  Implemented checks to save only when synthetic data was successfully generated.
  ```python
  if "synthetic_data" in result and result["synthetic_data"] is not None:
      output_file = f"{result['dataset']}_synthetic_data.csv"
      result["synthetic_data"].to_csv(output_file, index=False)
  else:
      logging.warning(f"Failed to generate synthetic data for {result['dataset']}")
  ```

### 5. **Configuration Loading and Parameterization Errors**
- **Error:**  
  Missing hyperparameters in `config.json` led to unexpected behaviors or crashes.

- **Fix:**  
  Provided default values using `kwargs.get()` and validated configuration keys.
  ```python
  config = load_config()
  ```

### 6. **Incorrect Identification of Target Variable**
- **Error:**  
  Assumed the target variable was always named 'Category', causing misalignment during data processing.

- **Fix:**  
  Dynamically identified the target variable based on dataset structure.
  ```python
  y = pd.Series(dataset.target, name="Category")
  ```

### 7. **Handling Numerical Targets During Evaluation**
- **Error:**  
  Applied classification metrics to numerical targets, leading to inappropriate evaluations.

- **Fix:**  
  Implemented conditional metric calculations based on target type.
  ```python
  if y_real.dtype == 'object' or y_real.dtype.name == 'category':
      # Classification metrics
  else:
      # Regression metrics
  ```

### 8. **Incorrect Conditional Vector Dimension Calculation**
- **Error:**  
  Miscalculated conditional vector dimensions causing model errors.

- **Fix:**  
  Accurately calculated `cond_dim` and mapped categorical columns correctly.
  ```python
  self.cond_dim = sum(self.discrete_column_category_counts)
  ```

---

## `benchmark_ctgan.py`

### 1. **Incorrect Sampling of Gaussian Components**
- **Error:**  
  Synthetic data generation sometimes sampled Gaussian components inaccurately, leading to unrealistic distributions.

- **Fix:**  
  Ensured correct probability sampling and filtered inactive components.
  ```python
  components = np.array([
      np.random.choice(len(p), p=p) if p.sum() > 0 else np.random.randint(len(p))
      for p in probs
  ])
  component_one_hot = component_one_hot[:, active_components]
  ```

### 2. **Incorrect Calculation of Evaluation Metrics**
- **Error:**  
  Metrics were inaccurately calculated, leading to misleading evaluations of synthetic data quality.

- **Fix:**  
  Validated metric implementations and handled edge cases.
  ```python
  "JSD_mean": np.mean(js_divergences) if js_divergences else None,
  "Wasserstein_mean": np.mean(wasserstein_distances) if wasserstein_distances else None
  ```

### 3. **Applying Classification Metrics to Numerical Targets**
- **Error:**  
  Used classification metrics (Accuracy, F1 Score) for regression tasks, leading to inappropriate evaluations.

- **Fix:**  
  Separated evaluation logic based on target variable type, applying appropriate metrics.
  ```python
  if y_real.dtype == 'object' or y_real.dtype.name == 'category':
      # Calculate Accuracy and F1 Score
      accuracy = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average='weighted')
  else:
      # Calculate R-squared
      r2 = r2_score(y_test, y_pred)
  ```

### 4. **Gradient Penalty Calculation Errors for WGAN-GP**
- **Error:**  
  Incorrect gradient penalty calculation led to unstable training.

- **Fix:**  
  Correctly computed gradients and integrated gradient penalty into discriminator loss.
  ```python
  gradients = torch.autograd.grad(
      outputs=validity_interpolates,
      inputs=interpolates,
      grad_outputs=torch.ones(validity_interpolates.size(), device=device),
      create_graph=True,
      retain_graph=True,
      only_inputs=True
  )[0]
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
  loss_d = loss_adv + gp + class_loss if self.num_classes > 0 else loss_adv + gp
  ```

### 5. **Handling Outliers and Anomalous Data Points**
- **Error:**  
  Outliers or anomalous data points adversely affected the training process, leading to unstable GAN training.

- **Fix:**  
  Applied data clipping and used robust scalers to mitigate the impact of outliers.
  ```python
  normalized_values = np.clip(normalized_values, -0.99, 0.99)
  scaler = MinMaxScaler()
  ```

### 6. **Handling High-Dimensional Output from DataTransformer**
- **Error:**  
  High-dimensional transformed data caused memory constraints and performance issues.

- **Fix:**  
  Limited the number of GMM components and applied dimensionality reduction techniques where necessary.
  ```python
  self.max_clusters = max_clusters
  active_components = vgm.weights_ > 1e-3
  component_one_hot = component_one_hot[:, active_components]
  ```

---

# Summary

By addressing these errors and implementing the corresponding fixes across the **`ctgan_adapter.py`**, **`run_ctgan.py`**, and **`benchmark_ctgan.py`** scripts, the CTGAN-based synthetic data generation pipeline has been made robust and reliable. The systematic handling of data preprocessing, model training, evaluation, and visualization ensures high-quality synthetic data that mirrors the statistical properties of real datasets while maintaining practical utility for downstream machine learning tasks.



### References

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems*, *32*.
```