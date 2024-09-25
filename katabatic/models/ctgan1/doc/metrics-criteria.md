# CTGAN Metrics Sheet

This metrics sheet details the evaluation metrics utilized in my CTGAN implementation. These metrics are inspired by two key papers:

1. **Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).** *Modeling tabular data using conditional GAN*. Advances in Neural Information Processing Systems, 32.

2. **Zhang, Y., Zaidi, N.A., Zhou, J., & Li, G. (2021, December).** *GANBLR: A Tabular Data Generation Model*. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 181-190). IEEE.

---

## `ctgan_adapter.py`

### 1. **Likelihood Fitness Metrics**

These metrics assess how well the synthetic data matches the underlying distribution of the real data.

#### **a. Lsyn (Log-Likelihood of Synthetic Data)**
- **Description:**  
  Measures the likelihood of the synthetic data under the Gaussian Mixture Model (GMM) trained on real data. Higher values indicate that synthetic data closely follows the learned distribution.
  
- **Inspiration:**  
  Inspired by Xu et al.'s use of log-likelihood to assess how well synthetic data captures the real data distribution.
  
- **Implementation:**  
  ```python
  from sklearn.mixture import GaussianMixture

  # Train GMM on real data
  gmm_real = GaussianMixture(n_components=10, random_state=42)
  gmm_real.fit(real_data)

  # Compute log-likelihood of synthetic data
  lsyn = gmm_real.score(synthetic_data)
  print(f"Lsyn (Log-Likelihood of Synthetic Data): {lsyn}")
  ```

#### **b. Ltest (Log-Likelihood of Test Data)**
- **Description:**  
  Evaluates the likelihood of real test data under a GMM retrained on the synthetic data. This metric helps in detecting overfitting by ensuring that the synthetic data generalizes well to unseen real data.
  
- **Inspiration:**  
  Addresses Xu et al.'s concern about overfitting models to synthetic data by introducing a secondary likelihood measure.
  
- **Implementation:**  
  ```python
  from sklearn.mixture import GaussianMixture

  # Train GMM on synthetic data
  gmm_synthetic = GaussianMixture(n_components=10, random_state=42)
  gmm_synthetic.fit(synthetic_data)

  # Compute log-likelihood of real test data
  ltest = gmm_synthetic.score(real_test_data)
  print(f"Ltest (Log-Likelihood of Test Data): {ltest}")
  ```

### 2. **Machine Learning Efficacy Metrics**

These metrics evaluate the usefulness of synthetic data for downstream machine learning tasks, ensuring that models trained on synthetic data perform comparably to those trained on real data.

#### **a. Accuracy**
- **Description:**  
  The proportion of correct predictions made by a classifier trained on synthetic data and evaluated on real test data.
  
- **Inspiration:**  
  Aligns with both Xu et al. and Zhang et al.'s evaluation of machine learning efficacy to determine the practical usefulness of synthetic data.
  
- **Implementation:**  
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  # Train classifier on synthetic data
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(synthetic_data_features, synthetic_data_labels)

  # Predict on real test data
  predictions = clf.predict(real_test_data_features)

  # Calculate accuracy
  accuracy = accuracy_score(real_test_data_labels, predictions)
  print(f"Accuracy: {accuracy}")
  ```

#### **b. F1 Score**
- **Description:**  
  The harmonic mean of precision and recall for classification tasks, providing a balance between the two.
  
- **Inspiration:**  
  Offers a balanced measure of classifier performance, especially useful in datasets with class imbalances as highlighted by both papers.
  
- **Implementation:**  
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import f1_score

  # Train classifier on synthetic data
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(synthetic_data_features, synthetic_data_labels)

  # Predict on real test data
  predictions = clf.predict(real_test_data_features)

  # Calculate F1 Score
  f1 = f1_score(real_test_data_labels, predictions, average='weighted')
  print(f"F1 Score: {f1}")
  ```

#### **c. R² Score**
- **Description:**  
  Measures the proportion of variance in the dependent variable that is predictable from the independent variables in regression tasks.
  
- **Inspiration:**  
  Evaluates the performance of regression models trained on synthetic data, as discussed in both papers' machine learning efficacy sections.
  
- **Implementation:**  
  ```python
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import r2_score

  # Train regression model on synthetic data
  reg = LinearRegression()
  reg.fit(synthetic_data_features, synthetic_data_target)

  # Predict on real validation data
  predictions = reg.predict(real_validation_data_features)

  # Calculate R² Score
  r2 = r2_score(real_validation_data_target, predictions)
  print(f"R² Score: {r2}")
  ```

### 3. **Statistical Similarity Metrics**

These metrics quantify the statistical resemblance between real and synthetic data distributions, ensuring that synthetic data mirrors the real data's properties.

#### **a. Jensen-Shannon Divergence (JSD)**
- **Description:**  
  Measures the similarity between two probability distributions. Lower values indicate higher similarity.
  
- **Inspiration:**  
  Used in both papers to evaluate distributional similarity between real and synthetic data.
  
- **Implementation:**  
  ```python
  from scipy.spatial.distance import jensenshannon
  import numpy as np

  def compute_jsd(real_data, synthetic_data):
      # Compute probability distributions
      real_dist, _ = np.histogram(real_data, bins=50, range=(real_data.min(), real_data.max()), density=True)
      synthetic_dist, _ = np.histogram(synthetic_data, bins=50, range=(real_data.min(), real_data.max()), density=True)

      # Normalize distributions
      real_dist = real_dist / real_dist.sum()
      synthetic_dist = synthetic_dist / synthetic_dist.sum()

      # Calculate JSD
      jsd = jensenshannon(real_dist, synthetic_dist)
      return jsd

  # Example usage for a specific feature
  feature_jsd = compute_jsd(real_data['feature_name'], synthetic_data['feature_name'])
  print(f"Jensen-Shannon Divergence for feature_name: {feature_jsd}")
  ```

#### **b. Wasserstein Distance (WD)**
- **Description:**  
  Computes the distance between two probability distributions, emphasizing differences in distribution shapes.
  
- **Inspiration:**  
  Provides a robust measure of distributional differences, as utilized in both papers to assess the quality of synthetic data.
  
- **Implementation:**  
  ```python
  from scipy.stats import wasserstein_distance

  def compute_wd(real_data, synthetic_data):
      wd = wasserstein_distance(real_data, synthetic_data)
      return wd

  # Example usage for a specific feature
  feature_wd = compute_wd(real_data['feature_name'], synthetic_data['feature_name'])
  print(f"Wasserstein Distance for feature_name: {feature_wd}")
  ```

### 4. **Interpretability Metrics**

These metrics assess the interpretability of the synthetic data generation process, ensuring that the model provides insights into feature importance and interactions.

#### **a. Local Interpretability (Using LIME)**
- **Description:**  
  Evaluates how individual synthetic data points can be interpreted in terms of feature contributions to their class labels.
  
- **Inspiration:**  
  Inspired by Zhang et al.'s focus on model interpretability, ensuring that synthetic data generation is transparent and explainable.
  
- **Implementation:**  
  ```python
  from lime import lime_tabular
  from sklearn.ensemble import RandomForestClassifier

  # Train classifier on synthetic data
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(synthetic_data_features, synthetic_data_labels)

  # Initialize LIME explainer
  explainer = lime_tabular.LimeTabularExplainer(
      training_data=synthetic_data_features.values,
      feature_names=synthetic_data_features.columns,
      class_names=['Class_0', 'Class_1'],
      mode='classification'
  )

  # Select a synthetic sample to explain
  sample = synthetic_data_features.iloc[0].values
  explanation = explainer.explain_instance(sample, clf.predict_proba, num_features=5)

  # Display explanation
  explanation.show_in_notebook()
  ```

#### **b. Global Interpretability**
- **Description:**  
  Assesses the overall impact of each feature on the synthetic data generation process, identifying which features most influence class labels.
  
- **Inspiration:**  
  Aligns with Zhang et al.'s emphasis on understanding feature importance globally, providing insights into how features interact during data generation.
  
- **Implementation:**  
  ```python
  from sklearn.ensemble import RandomForestClassifier
  import pandas as pd
  import matplotlib.pyplot as plt

  # Train classifier on synthetic data
  clf = RandomForestClassifier(n_estimators=100, random_state=42)
  clf.fit(synthetic_data_features, synthetic_data_labels)

  # Get feature importances
  importances = clf.feature_importances_
  feature_names = synthetic_data_features.columns
  feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

  # Plot feature importances
  plt.figure(figsize=(10, 6))
  feature_importance.plot(kind='bar')
  plt.title('Global Feature Importances')
  plt.ylabel('Importance Score')
  plt.xlabel('Features')
  plt.tight_layout()
  plt.show()
  ```

---

## `run_ctgan.py`

### 1. **Configuration Validation Metrics**

Ensuring that the CTGAN is configured correctly is crucial for reliable training and evaluation.

#### **a. Configuration Key Validation**
- **Description:**  
  Checks for the presence of essential keys in the configuration file (`config.json`) to prevent runtime errors and ensure all necessary parameters are set.
  
- **Inspiration:**  
  Inspired by Xu et al.'s emphasis on robust benchmarking and consistent evaluation across different setups.
  
- **Implementation:**  
  ```python
  import json
  import sys

  REQUIRED_KEYS = ['batch_size', 'epochs', 'generator_layers', 'discriminator_layers', 'latent_dim']

  def validate_config(config_path):
      with open(config_path, 'r') as file:
          config = json.load(file)
      
      missing_keys = [key for key in REQUIRED_KEYS if key not in config]
      if missing_keys:
          print(f"Missing configuration keys: {missing_keys}")
          sys.exit(1)
      else:
          print("Configuration validation passed.")

  # Example usage
  validate_config('config.json')
  ```

### 2. **Synthetic Data Generation Metrics**

Evaluates the quality and consistency of the synthetic data generated by CTGAN.

#### **a. Sample Generation Consistency**
- **Description:**  
  Ensures that the number of generated samples matches the required size, preventing indexing errors and maintaining data integrity.
  
- **Inspiration:**  
  Maintains the fidelity of synthetic data as emphasized in Xu et al.'s and Zhang et al.'s benchmarking processes.
  
- **Implementation:**  
  ```python
  def generate_synthetic_data(ctgan_model, target_size):
      synthetic_data = ctgan_model.sample(target_size)
      actual_size = len(synthetic_data)
      assert actual_size == target_size, f"Expected {target_size} samples, but got {actual_size}"
      print(f"Sample Generation Consistency Check Passed: {actual_size} samples generated.")
      return synthetic_data

  # Example usage
  synthetic_data = generate_synthetic_data(ctgan, target_size=1000)
  ```

#### **b. Conditional Vector Accuracy**
- **Description:**  
  Verifies that conditional vectors correctly represent the desired conditions, ensuring that synthetic data adheres to specified categories or classes.
  
- **Inspiration:**  
  Aligns with Xu et al.'s and Zhang et al.'s focus on conditional generation to handle imbalanced categorical data effectively.
  
- **Implementation:**  
  ```python
  import numpy as np

  def validate_conditional_vectors(synthetic_data, conditions):
      # Assuming 'condition' is a column in the synthetic_data
      condition_matches = synthetic_data['condition'].isin(conditions).all()
      if condition_matches:
          print("Conditional Vector Accuracy Check Passed.")
      else:
          print("Conditional Vector Accuracy Check Failed.")
          missing_conditions = set(conditions) - set(synthetic_data['condition'].unique())
          print(f"Missing conditions: {missing_conditions}")
          sys.exit(1)

  # Example usage
  desired_conditions = ['A', 'B', 'C']
  validate_conditional_vectors(synthetic_data, desired_conditions)
  ```

---

## `benchmark_ctgan.py`

### 1. **Evaluation Framework Metrics**

Implements a comprehensive evaluation framework inspired by both Xu et al. and Zhang et al. to assess synthetic data quality across multiple dimensions.

#### **a. Likelihood Fitness (Lsyn and Ltest)**
- **Description:**  
  - **Lsyn:** Evaluates the likelihood of synthetic data under the real data's distribution.
  - **Ltest:** Assesses the likelihood of real test data under a model trained on synthetic data.
  
- **Inspiration:**  
  Directly adopted from Xu et al. to measure how well synthetic data captures the underlying distribution and generalizes to new data.
  
- **Implementation:**  
  ```python
  from sklearn.mixture import GaussianMixture

  def compute_lsyn(real_data, synthetic_data, n_components=10):
      gmm_real = GaussianMixture(n_components=n_components, random_state=42)
      gmm_real.fit(real_data)
      lsyn = gmm_real.score(synthetic_data)
      return lsyn

  def compute_ltest(synthetic_data, real_test_data, n_components=10):
      gmm_synthetic = GaussianMixture(n_components=n_components, random_state=42)
      gmm_synthetic.fit(synthetic_data)
      ltest = gmm_synthetic.score(real_test_data)
      return ltest

  # Example usage
  lsyn = compute_lsyn(real_data, synthetic_data)
  ltest = compute_ltest(synthetic_data, real_test_data)
  print(f"Lsyn: {lsyn}, Ltest: {ltest}")
  ```

#### **b. Machine Learning Efficacy (TSTR and TRTR)**
- **Description:**  
  - **TSTR (Train on Synthetic, Test on Real):** Measures the performance of machine learning models trained on synthetic data and tested on real data.
  - **TRTR (Train on Real, Test on Real):** Measures the performance of machine learning models trained and tested on real data, serving as an upper benchmark.
  
- **Inspiration:**  
  Adopted from Zhang et al.'s GANBLR paper to evaluate the practical utility of synthetic data for machine learning tasks.
  
- **Implementation:**  
  ```python
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score, f1_score, r2_score

  def evaluate_tstr(synthetic_features, synthetic_labels, real_test_features, real_test_labels):
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      clf.fit(synthetic_features, synthetic_labels)
      predictions = clf.predict(real_test_features)
      accuracy = accuracy_score(real_test_labels, predictions)
      f1 = f1_score(real_test_labels, predictions, average='weighted')
      return accuracy, f1

  def evaluate_trtr(real_train_features, real_train_labels, real_test_features, real_test_labels):
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      clf.fit(real_train_features, real_train_labels)
      predictions = clf.predict(real_test_features)
      accuracy = accuracy_score(real_test_labels, predictions)
      f1 = f1_score(real_test_labels, predictions, average='weighted')
      return accuracy, f1

  # Example usage
  tstr_accuracy, tstr_f1 = evaluate_tstr(synthetic_data_features, synthetic_data_labels, real_test_data_features, real_test_data_labels)
  trtr_accuracy, trtr_f1 = evaluate_trtr(real_train_data_features, real_train_data_labels, real_test_data_features, real_test_data_labels)
  print(f"TSTR Accuracy: {tstr_accuracy}, TSTR F1 Score: {tstr_f1}")
  print(f"TRTR Accuracy: {trtr_accuracy}, TRTR F1 Score: {trtr_f1}")
  ```

#### **c. Statistical Similarity (JSD and WD)**
- **Description:**  
  Quantifies the statistical resemblance between real and synthetic data distributions using Jensen-Shannon Divergence and Wasserstein Distance.
  
- **Inspiration:**  
  Both Xu et al. and Zhang et al. utilize these metrics to ensure that synthetic data mirrors real data's statistical properties.
  
- **Implementation:**  
  ```python
  from scipy.spatial.distance import jensenshannon
  from scipy.stats import wasserstein_distance
  import numpy as np

  def compute_jsd(real_data, synthetic_data):
      real_dist, _ = np.histogram(real_data, bins=50, range=(real_data.min(), real_data.max()), density=True)
      synthetic_dist, _ = np.histogram(synthetic_data, bins=50, range=(real_data.min(), real_data.max()), density=True)
      real_dist = real_dist / real_dist.sum()
      synthetic_dist = synthetic_dist / synthetic_dist.sum()
      jsd = jensenshannon(real_dist, synthetic_dist)
      return jsd

  def compute_wd(real_data, synthetic_data):
      wd = wasserstein_distance(real_data, synthetic_data)
      return wd

  # Example usage for a specific feature
  feature_jsd = compute_jsd(real_data['feature_name'], synthetic_data['feature_name'])
  feature_wd = compute_wd(real_data['feature_name'], synthetic_data['feature_name'])
  print(f"JSD for feature_name: {feature_jsd}, WD for feature_name: {feature_wd}")
  ```

#### **d. Interpretability Metrics**
- **Description:**  
  Assesses the interpretability of the synthetic data generation process, focusing on both local and global feature importance.
  
- **Inspiration:**  
  Derived from Zhang et al.'s GANBLR paper, emphasizing the importance of interpretability in synthetic data models.
  
- **Implementation:**  
  ```python
  from lime import lime_tabular
  from sklearn.ensemble import RandomForestClassifier
  import pandas as pd
  import matplotlib.pyplot as plt

  def compute_local_interpretability(synthetic_data_features, synthetic_data_labels):
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      clf.fit(synthetic_data_features, synthetic_data_labels)
      explainer = lime_tabular.LimeTabularExplainer(
          training_data=synthetic_data_features.values,
          feature_names=synthetic_data_features.columns,
          class_names=['Class_0', 'Class_1'],
          mode='classification'
      )
      sample = synthetic_data_features.iloc[0].values
      explanation = explainer.explain_instance(sample, clf.predict_proba, num_features=5)
      explanation.show_in_notebook()

  def compute_global_interpretability(synthetic_data_features, synthetic_data_labels):
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      clf.fit(synthetic_data_features, synthetic_data_labels)
      importances = clf.feature_importances_
      feature_names = synthetic_data_features.columns
      feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
      plt.figure(figsize=(10, 6))
      feature_importance.plot(kind='bar')
      plt.title('Global Feature Importances')
      plt.ylabel('Importance Score')
      plt.xlabel('Features')
      plt.tight_layout()
      plt.show()

  # Example usage
  compute_local_interpretability(synthetic_data_features, synthetic_data_labels)
  compute_global_interpretability(synthetic_data_features, synthetic_data_labels)
  ```

### 2. **Benchmarking Results Aggregation**

Aggregates and compares results across different datasets and metrics to provide a holistic view of CTGAN's performance.

#### **a. Ranking and Scoring**
- **Description:**  
  Ranks algorithms based on their performance across multiple metrics and datasets, providing an average performance score for each method.
  
- **Inspiration:**  
  Ensures a fair and comprehensive comparison as outlined in both papers' benchmarking systems.
  
- **Implementation:**  
  ```python
  import pandas as pd

  def rank_algorithms(results_df):
      # Assuming results_df has algorithms as rows and metrics as columns
      results_df['Average_Score'] = results_df.mean(axis=1)
      ranked_df = results_df.sort_values(by='Average_Score', ascending=False)
      return ranked_df

  # Example usage
  data = {
      'Algorithm': ['CTGAN', 'Baseline1', 'Baseline2'],
      'Accuracy': [0.85, 0.80, 0.78],
      'F1_Score': [0.83, 0.79, 0.75],
      'R2_Score': [0.80, 0.76, 0.74],
      'JSD': [0.05, 0.07, 0.08],
      'WD': [0.10, 0.15, 0.20]
  }
  results_df = pd.DataFrame(data).set_index('Algorithm')
  ranked_results = rank_algorithms(results_df)
  print(ranked_results)
  ```

#### **b. Performance Reporting**
- **Description:**  
  Summarizes the performance of CTGAN against baseline models across various metrics, facilitating easy comparison.
  
- **Inspiration:**  
  Facilitates comprehensive benchmarking as presented in both papers, enabling identification of strengths and weaknesses.
  
- **Implementation:**  
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt

  def generate_performance_report(results_df):
      metrics = results_df.columns[:-1]  # Exclude 'Average_Score'
      for metric in metrics:
          results_df[metric].plot(kind='bar', figsize=(8,6))
          plt.title(f'Performance Comparison: {metric}')
          plt.ylabel(metric)
          plt.xlabel('Algorithm')
          plt.tight_layout()
          plt.show()

  # Example usage
  generate_performance_report(ranked_results)
  ```

---

# Summary

In developing my CTGAN implementation, I incorporated a suite of evaluation metrics inspired by both the papers "Modeling Tabular Data using Conditional GAN" by Xu et al. (2019) and "GANBLR: A Tabular Data Generation Model" by Zhang et al. (2021). These metrics encompass:

- **Likelihood Fitness:**  
  Assessing how well synthetic data captures the real data distribution using Lsyn and Ltest.

- **Machine Learning Efficacy:**  
  Evaluating the practical utility of synthetic data for training machine learning models through Accuracy, F1 Score, R² Score, and frameworks like TSTR and TRTR.

- **Statistical Similarity:**  
  Quantifying the statistical resemblance between real and synthetic data using Jensen-Shannon Divergence (JSD) and Wasserstein Distance (WD).

- **Interpretability:**  
  Ensuring that the synthetic data generation process is transparent and explainable via local and global interpretability metrics, leveraging tools like LIME.

By organizing these metrics across the **`ctgan_adapter.py`**, **`run_ctgan.py`**, and **`benchmark_ctgan.py`** scripts, I established a structured and comprehensive evaluation framework. This alignment with the rigorous benchmarking standards set forth in both referenced papers ensures that the synthetic data generated by CTGAN is both high-quality and practically useful for downstream applications.

---

# References

- Zhang, Y., Zaidi, N.A., Zhou, J. and Li, G., 2021, December. GANBLR: a tabular data generation model. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 181-190). IEEE.

- Xu, L., Skoularidou, M., Cuesta-Infante, A. and Veeramachaneni, K., 2019. Modeling tabular data using conditional gan. Advances in neural information processing systems, 32.