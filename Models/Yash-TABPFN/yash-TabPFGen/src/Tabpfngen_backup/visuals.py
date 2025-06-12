import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import Counter
from scipy.stats import probplot

def visualize_classification_results(X_orig, y_orig, X_synth, y_synth, feature_names=None):
    """
    Comprehensive visualization of original vs synthetic classification data
    
    Args:
        X_orig: Original features
        y_orig: Original labels
        X_synth: Synthetic labels
        y_synth: Synthetic labels
        feature_names: Optional list of feature names
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Class Distribution Comparison
    plt.subplot(2, 2, 1)
    orig_counts = Counter(y_orig)
    synth_counts = Counter(y_synth)
    
    classes = sorted(list(set(y_orig) | set(y_synth)))
    width = 0.35
    plt.bar(np.array(classes) - width/2, 
           [orig_counts[c] for c in classes], 
           width, 
           label='Original', 
           alpha=0.6,
           color='blue')
    plt.bar(np.array(classes) + width/2, 
           [synth_counts[c] for c in classes], 
           width, 
           label='Synthetic', 
           alpha=0.6,
           color='red')
    plt.title('Class Distribution Comparison', fontsize=12, pad=20)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. t-SNE Visualization
    plt.subplot(2, 2, 2)
    
    # Combine and scale data
    X_combined = np.vstack([X_orig, X_synth])
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined_scaled)
    
    # Split back into original and synthetic
    n_orig = len(X_orig)
    X_tsne_orig = X_tsne[:n_orig]
    X_tsne_synth = X_tsne[n_orig:]
    
    # Plot with different colors for each class
    for c in classes:
        # Original data
        mask_orig = y_orig == c
        plt.scatter(X_tsne_orig[mask_orig, 0], X_tsne_orig[mask_orig, 1], 
                   alpha=0.6, marker='o', label=f'Original Class {c}')
        # Synthetic data
        mask_synth = y_synth == c
        plt.scatter(X_tsne_synth[mask_synth, 0], X_tsne_synth[mask_synth, 1], 
                   alpha=0.6, marker='x', label=f'Synthetic Class {c}')
    
    plt.title('t-SNE Visualization', fontsize=12, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    plt.subplot(2, 2, 3)
    
    # Select top features based on random forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_orig, y_orig)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_orig.shape[1])]
    
    # Get top 5 important features
    importance = rf.feature_importances_
    top_features_idx = importance.argsort()[-5:][::-1]
    
    # Plot feature importance
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    plt.bar(range(5), importance[top_features_idx], color=colors)
    plt.title('Top 5 Important Features', fontsize=12, pad=20)
    plt.xticks(range(5), [feature_names[i] for i in top_features_idx], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Feature Distributions for Top Feature
    plt.subplot(2, 2, 4)
    top_feature_idx = top_features_idx[0]
    
    # Plot distributions for each class
    for i, label in enumerate(classes):
        # Original data
        orig_data = X_orig[y_orig == label][:, top_feature_idx]
        plt.hist(orig_data, bins=20, alpha=0.3, 
                label=f'Original Class {label}', 
                color=plt.cm.Set3(i), density=True)
        
        # Synthetic data
        synth_data = X_synth[y_synth == label][:, top_feature_idx]
        plt.hist(synth_data, bins=20, alpha=0.3, 
                label=f'Synthetic Class {label}', 
                color=plt.cm.Set3(i + len(classes)), density=True)
    
    plt.title(f'Distribution of Top Feature:\n{feature_names[top_feature_idx]}', 
              fontsize=12, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional Plot: Feature Correlations
    plt.figure(figsize=(12, 6))
    
    # Original data correlations
    plt.subplot(1, 2, 1)
    orig_corr = np.corrcoef(X_orig.T)
    plt.imshow(orig_corr, cmap='coolwarm', aspect='equal')
    plt.colorbar()
    plt.title('Original Feature Correlations', fontsize=12, pad=20)
    
    # Synthetic data correlations
    plt.subplot(1, 2, 2)
    synth_corr = np.corrcoef(X_synth.T)
    plt.imshow(synth_corr, cmap='coolwarm', aspect='equal')
    plt.colorbar()
    plt.title('Synthetic Feature Correlations', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    
    # Initialize generator
    generator = TabPFGenV2(n_sgld_steps=500)
    
    # Generate synthetic data
    X_synth, y_synth = generator.generate_classification(
        X, y,
        n_samples=100,
        balance_classes=True
    )
    
    # Visualize results
    visualize_classification_results(
        X, y, X_synth, y_synth,
        feature_names=load_breast_cancer().feature_names
    )



def visualize_regression_results(X_orig, y_orig, X_synth, y_synth, feature_names=None):
    """
    Comprehensive visualization of original vs synthetic regression data
    
    Args:
        X_orig: Original features
        y_orig: Original target values
        X_synth: Synthetic features
        y_synth: Synthetic target values
        feature_names: Optional list of feature names
    """
    # First plot: Basic distribution comparisons
    plt.figure(figsize=(15, 5))
    
    # Original vs Synthetic Distribution
    plt.subplot(1, 3, 1)
    plt.hist(y_orig, bins=30, alpha=0.5, label='Original', density=True, color='blue')
    plt.hist(y_synth, bins=30, alpha=0.5, label='Synthetic', density=True, color='red')
    plt.title('Distribution Comparison')
    plt.xlabel('Target Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Q-Q Plot
    plt.subplot(1, 3, 2)
    probplot(y_synth, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Synthetic Data')
    plt.grid(True, alpha=0.3)

    # Box Plot Comparison
    plt.subplot(1, 3, 3)
    plt.boxplot([y_orig, y_synth], labels=['Original', 'Synthetic'])
    plt.title('Box Plot Comparison')
    plt.ylabel('Target Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Second plot: Feature importance and relationships
    plt.figure(figsize=(15, 5))

    # Feature Importance
    plt.subplot(1, 3, 1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_orig, y_orig)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_orig.shape[1])]
    
    # Get top 5 important features
    importance = rf.feature_importances_
    top_features_idx = importance.argsort()[-5:][::-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    plt.bar(range(5), importance[top_features_idx], color=colors)
    plt.title('Top 5 Important Features')
    plt.xticks(range(5), [feature_names[i] for i in top_features_idx], rotation=45)
    plt.grid(True, alpha=0.3)

    # Most Important Feature vs Target
    plt.subplot(1, 3, 2)
    top_feature_idx = top_features_idx[0]
    
    plt.scatter(X_orig[:, top_feature_idx], y_orig, 
               alpha=0.5, label='Original', color='blue')
    plt.scatter(X_synth[:, top_feature_idx], y_synth, 
               alpha=0.5, label='Synthetic', color='red')
    plt.xlabel(f'Top Feature: {feature_names[top_feature_idx]}')
    plt.ylabel('Target Value')
    plt.title('Top Feature vs Target')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # t-SNE Visualization
    plt.subplot(1, 3, 3)
    X_combined = np.vstack([X_orig, X_synth])
    y_combined = np.hstack([y_orig, y_synth])
    
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined_scaled)
    
    # Color points by target value
    plt.scatter(X_tsne[:len(X_orig), 0], X_tsne[:len(X_orig), 1], 
               c=y_orig, cmap='viridis', alpha=0.5, label='Original')
    plt.scatter(X_tsne[len(X_orig):, 0], X_tsne[len(X_orig):, 1], 
               c=y_synth, cmap='viridis', alpha=0.5, marker='x', label='Synthetic')
    plt.colorbar(label='Target Value')
    plt.title('t-SNE Visualization')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Third plot: Additional statistical analysis
    plt.figure(figsize=(10, 5))

    # Residuals Distribution
    plt.subplot(1, 2, 1)
    residuals_orig = y_orig - np.mean(y_orig)
    residuals_synth = y_synth - np.mean(y_synth)
    
    plt.hist(residuals_orig, bins=30, alpha=0.5, label='Original', density=True, color='blue')
    plt.hist(residuals_synth, bins=30, alpha=0.5, label='Synthetic', density=True, color='red')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Target Value Range Coverage
    plt.subplot(1, 2, 2)
    percentiles = np.linspace(0, 100, 20)
    orig_percentiles = np.percentile(y_orig, percentiles)
    synth_percentiles = np.percentile(y_synth, percentiles)

    plt.plot(percentiles, orig_percentiles, 'b-', label='Original')
    plt.plot(percentiles, synth_percentiles, 'r--', label='Synthetic')
    plt.title('Target Value Range Coverage')
    plt.xlabel('Percentile')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Fourth plot: Additional statistical analysis
    plt.figure(figsize=(12, 6))

    # Feature Correlations Comparison
    orig_corr = np.corrcoef(X_orig.T)
    synth_corr = np.corrcoef(X_synth.T)
    #corr_diff = np.abs(orig_corr - synth_corr)
    plt.subplot(1, 2, 1)
    plt.imshow(orig_corr, cmap='coolwarm', aspect='equal')
    plt.colorbar(label='Correlation Value')
    plt.title('Original Feature Correlations', fontsize=12, pad=20)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')

    plt.subplot(1, 2, 2)
    plt.imshow(synth_corr, cmap='coolwarm', aspect='equal')
    plt.colorbar(label='Correlation Value')
    plt.title('Synthetic Feature Correlations', fontsize=12, pad=20)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    
    # Load regression dataset
    X, y = load_diabetes(return_X_y=True)
    
    # Initialize generator
    generator = TabPFGenV2(n_sgld_steps=500)
    
    # Generate synthetic regression data
    X_synth, y_synth = generator.generate_regression(
        X, y,
        n_samples=100,
        use_quantiles=True
    )
    
    # Visualize results
    visualize_regression_results(
        X, y, X_synth, y_synth,
        feature_names=load_diabetes().feature_names
    )
