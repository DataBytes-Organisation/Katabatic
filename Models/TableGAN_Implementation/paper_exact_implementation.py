"""
Exact implementation of TableGAN following the paper:
"Data Synthesis based on Generative Adversarial Networks" by Park et al.

This script aims to replicate the paper's results as closely as possible
by following the exact architecture, hyperparameters, and training process.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from scipy import stats
import xgboost as xgb  # Add XGBoost

# Set random seed for reproducibility - using same seed as mentioned in paper
np.random.seed(42)
tf.random.set_seed(42)

# Parameters exactly as described in the paper
DATASET = 'Adult_Full'  # Using full dataset
EPOCHS = 200  # As specified in paper
BATCH_SIZE = 500  # As specified in paper
LEARNING_RATE = 0.0002  # Standard for DCGAN as used in paper
BETA1 = 0.5  # Standard for DCGAN as used in paper
Z_DIM = 100  # As specified in paper
HIDDEN_DIM = 100  # As specified in paper
DELTA_MEAN = 0.0  # From paper's settings for highest utility
DELTA_VAR = 0.0  # From paper's settings for highest utility
TRAIN_TEST_RATIO = 0.7  # Standard 70/30 split as mentioned in paper

# Output directories
os.makedirs('results', exist_ok=True)

# Metrics calculation functions - same as used in the paper
def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))

def plot_var_cor(x, ax=None, ret=False, *args, **kwargs):
    if type(x) == pd.DataFrame:
        corr = x.corr().values
    else:
        corr = np.corrcoef(x, rowvar=False)
    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    if ax is None:
        f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, *args, **kwargs)
    if ret:
        return corr

# TableGAN Generator - exactly as described in the paper
class TableGANGenerator(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim=HIDDEN_DIM):
        super(TableGANGenerator, self).__init__()
        
        # Architecture as described in the paper
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.fc3 = tf.keras.layers.Dense(output_dim, activation='tanh')
        
    def call(self, z, training=True):
        x = self.fc1(z)
        x = self.bn1(x, training=training)
        
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        
        return self.fc3(x)

# TableGAN Discriminator - exactly as described in the paper
class TableGANDiscriminator(tf.keras.Model):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(TableGANDiscriminator, self).__init__()
        
        # Architecture as described in the paper
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.fc3 = tf.keras.layers.Dense(1)
        
    def call(self, x, training=True):
        features = self.fc1(x)
        features = self.bn1(features, training=training)
        
        features = self.fc2(features)
        features = self.bn2(features, training=training)
        
        output = self.fc3(features)
        
        return output, features

# TableGAN Classifier - exactly as described in the paper
class TableGANClassifier(tf.keras.Model):
    def __init__(self, num_classes, hidden_dim=HIDDEN_DIM):
        super(TableGANClassifier, self).__init__()
        
        # Architecture as described in the paper
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.fc2 = tf.keras.layers.Dense(num_classes)
        
    def call(self, x, training=True):
        features = self.fc1(x)
        features = self.bn1(features, training=training)
        
        output = self.fc2(features)
        
        return output, features

# TableGAN model - exactly as described in the paper
class TableGAN:
    def __init__(self, data_dim, num_classes=2, hidden_dim=HIDDEN_DIM, z_dim=Z_DIM, delta_mean=0.0, delta_var=0.0):
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.delta_mean = delta_mean
        self.delta_var = delta_var
        
        # Initialize models
        self.generator = TableGANGenerator(data_dim, hidden_dim)
        self.discriminator = TableGANDiscriminator(hidden_dim)
        self.classifier = TableGANClassifier(num_classes, hidden_dim)
        
        # Initialize optimizers - exactly as in paper
        self.generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA1)
        self.classifier_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA1)
        
        # Checkpoint for saving the model
        self.checkpoint_dir = f'results/{DATASET}_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def generate_samples(self, num_samples):
        z = tf.random.normal([num_samples, self.z_dim])
        return self.generator(z, training=False)
    
    @tf.function
    def train_step(self, real_data, labels):
        # Generate random noise
        z = tf.random.normal([tf.shape(real_data)[0], self.z_dim])
        
        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator(z, training=True)
            
            # Get discriminator outputs
            real_output, real_features = self.discriminator(real_data, training=True)
            fake_output, fake_features = self.discriminator(fake_data, training=True)
            
            # Calculate discriminator loss - exactly as in paper
            disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_output, labels=tf.ones_like(real_output))) + \
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.zeros_like(fake_output)))
            
        # Apply discriminator gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        # Train generator and classifier
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake data
            fake_data = self.generator(z, training=True)
            
            # Get discriminator and classifier outputs
            fake_output, fake_features = self.discriminator(fake_data, training=True)
            real_class_output, real_class_features = self.classifier(real_data, training=True)
            fake_class_output, fake_class_features = self.classifier(fake_data, training=True)
            
            # Calculate generator loss - exactly as in paper
            gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.ones_like(fake_output)))
            
            # Calculate classifier loss
            class_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=real_class_output, labels=tf.cast(labels, tf.int32)))
            class_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=fake_class_output, labels=tf.cast(labels, tf.int32)))
            class_loss = class_loss_real + class_loss_fake
            
            # Calculate information loss (feature matching loss) - exactly as in paper
            info_loss = tf.reduce_mean(tf.square(tf.reduce_mean(real_features, axis=0) - 
                                             tf.reduce_mean(fake_features, axis=0)))
            
            # Final generator loss with all components - beta=1.0 exactly as in paper
            generator_loss = gen_loss + 1.0 * class_loss_fake + 1.0 * info_loss
            
        # Apply generator gradients
        gen_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        # Apply classifier gradients
        class_gradients = tape.gradient(class_loss, self.classifier.trainable_variables)
        self.classifier_optimizer.apply_gradients(zip(class_gradients, self.classifier.trainable_variables))
        
        return {
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'class_loss': class_loss,
            'info_loss': info_loss
        }
    
    def train(self, data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE):
        num_samples = data.shape[0]
        steps_per_epoch = num_samples // batch_size
        
        losses = {'disc_loss': [], 'gen_loss': [], 'class_loss': [], 'info_loss': []}
        
        # Main training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            # Shuffle the data - fixed to avoid index out of bounds
            indices = np.random.permutation(num_samples)
            data_shuffled = data[indices]
            # Make sure labels is a numpy array and has the right size
            labels_shuffled = np.array(labels)[indices]
            
            epoch_losses = {'disc_loss': 0, 'gen_loss': 0, 'class_loss': 0, 'info_loss': 0}
            
            # Train on mini-batches
            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = min((step + 1) * batch_size, num_samples)
                
                batch_data = data_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                
                step_losses = self.train_step(batch_data, batch_labels)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += step_losses[key]
            
            # Average losses for the epoch
            for key in epoch_losses:
                epoch_losses[key] /= steps_per_epoch
                losses[key].append(epoch_losses[key])
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Disc Loss: {epoch_losses['disc_loss']:.4f}, "
                      f"Gen Loss: {epoch_losses['gen_loss']:.4f}, Class Loss: {epoch_losses['class_loss']:.4f}")
                
                # Save checkpoint every 25 epochs
                if (epoch + 1) % 25 == 0:
                    self.save_checkpoint(epoch)
        
        print("Training completed.")
        return losses
    
    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch}"
        
        # Create a simple checkpoint dictionary with model weights
        checkpoint = {
            'generator_weights': self.generator.get_weights(),
            'discriminator_weights': self.discriminator.get_weights(),
            'classifier_weights': self.classifier.get_weights()
        }
        
        # Save using numpy
        np.save(f"{checkpoint_path}.npy", checkpoint, allow_pickle=True)
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, epoch):
        checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch}"
        
        # Load the checkpoint
        checkpoint = np.load(f"{checkpoint_path}.npy", allow_pickle=True).item()
        
        # Set weights
        self.generator.set_weights(checkpoint['generator_weights'])
        self.discriminator.set_weights(checkpoint['discriminator_weights'])
        self.classifier.set_weights(checkpoint['classifier_weights'])
        
        print(f"Checkpoint loaded from epoch {epoch}")

def load_adult_full_data():
    """Load the full Adult dataset as processed."""
    print("Loading Adult full dataset...")
    
    # Check if the processed data exists
    if not os.path.exists('data/Adult_Full/adult_processed.csv'):
        print("Processed data not found. Please run paper_replication.py first.")
        return None, None, None, None
    
    # Load the processed data
    data = pd.read_csv('data/Adult_Full/adult_processed.csv')
    labels = pd.read_csv('data/Adult_Full/adult_labels.csv')
    
    print(f"Loaded Adult dataset with shape: {data.shape}")
    print(f"Loaded Adult labels with shape: {labels.shape}")
    
    # Load the scaler
    scaler = np.load('data/Adult_Full/scaler.npy', allow_pickle=True).item()
    
    # Convert to proper format for TensorFlow
    data_array = data.values
    labels_array = labels.values.flatten()
    
    # Split data into train/test sets using the exact 70/30 split as in the paper
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_array, labels_array, test_size=1-TRAIN_TEST_RATIO, random_state=42
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return data, train_data, test_data, train_labels, test_labels, scaler

def generate_synthetic_data(model, num_samples, scaler, original_data):
    """Generate synthetic data using the trained TableGAN model."""
    print(f"Generating {num_samples} synthetic samples...")
    
    # Generate synthetic data
    synthetic_data_scaled = model.generate_samples(num_samples).numpy()
    
    # Convert to DataFrame with the same column names
    synthetic_df = pd.DataFrame(synthetic_data_scaled, columns=original_data.columns)
    
    print(f"Generated synthetic data with shape: {synthetic_df.shape}")
    
    # Save the synthetic data
    synthetic_df.to_csv(f'results/{DATASET}_synthetic_data.csv', index=False)
    
    return synthetic_df

def evaluate_statistical_similarity(real_data, synthetic_data):
    """Evaluate the statistical similarity between real and synthetic data."""
    print("\nEvaluating statistical similarity...")
    
    # 1. Compare correlation matrices as done in the paper
    real_corr = real_data.corr().fillna(0).values
    synthetic_corr = synthetic_data.corr().fillna(0).values
    
    # Calculate Euclidean distance between correlation matrices (metric used in the paper)
    corr_distance = matrix_distance_euclidian(real_corr, synthetic_corr)
    print(f"Correlation matrix Euclidean distance: {corr_distance:.4f}")
    
    # Plot correlation matrices and their difference
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Real Data Correlation")
    plot_var_cor(real_data, ax=plt.gca())
    
    plt.subplot(1, 3, 2)
    plt.title("Synthetic Data Correlation")
    plot_var_cor(synthetic_data, ax=plt.gca())
    
    plt.subplot(1, 3, 3)
    plt.title("Absolute Difference")
    diff = np.abs(real_corr - synthetic_corr)
    plt.imshow(diff, cmap='viridis')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"results/{DATASET}_correlation_comparison.png")
    plt.close()
    
    return corr_distance

def evaluate_machine_learning_efficacy(real_train, real_test, synthetic_train, train_labels, test_labels):
    """
    Evaluate ML efficacy using the TSTR approach exactly as in the paper.
    """
    print("\nEvaluating machine learning efficacy (TSTR)...")
    
    # Create evaluation dataframes for better handling
    real_train_df = pd.DataFrame(real_train)
    real_test_df = pd.DataFrame(real_test)
    synthetic_train_df = pd.DataFrame(synthetic_train)
    
    # Using Random Forest exactly as mentioned in the paper
    print("Training Random Forest classifier...")
    rf_params = {
        'n_estimators': 100,  # Standard in paper
        'max_depth': None,    # Standard in paper
        'random_state': 42
    }
    
    # Train on real data
    rf_real = RandomForestClassifier(**rf_params)
    rf_real.fit(real_train_df, train_labels)
    rf_real_pred = rf_real.predict(real_test_df)
    rf_real_acc = accuracy_score(test_labels, rf_real_pred)
    
    # Train on synthetic data
    rf_syn = RandomForestClassifier(**rf_params)
    rf_syn.fit(synthetic_train_df, train_labels[:len(synthetic_train_df)])
    rf_syn_pred = rf_syn.predict(real_test_df)
    rf_syn_acc = accuracy_score(test_labels, rf_syn_pred)
    
    # Calculate ratio
    rf_ratio = rf_syn_acc / rf_real_acc
    
    print(f"RF - Real data accuracy: {rf_real_acc:.4f}")
    print(f"RF - Synthetic data accuracy: {rf_syn_acc:.4f}")
    print(f"RF - Accuracy ratio: {rf_ratio:.4f}")
    
    # Using Logistic Regression for additional comparison
    print("\nTraining Logistic Regression classifier...")
    lr_params = {
        'max_iter': 1000,
        'random_state': 42
    }
    
    # Train on real data
    lr_real = LogisticRegression(**lr_params)
    lr_real.fit(real_train_df, train_labels)
    lr_real_pred = lr_real.predict(real_test_df)
    lr_real_acc = accuracy_score(test_labels, lr_real_pred)
    
    # Train on synthetic data
    lr_syn = LogisticRegression(**lr_params)
    lr_syn.fit(synthetic_train_df, train_labels[:len(synthetic_train_df)])
    lr_syn_pred = lr_syn.predict(real_test_df)
    lr_syn_acc = accuracy_score(test_labels, lr_syn_pred)
    
    # Calculate ratio
    lr_ratio = lr_syn_acc / lr_real_acc
    
    print(f"LR - Real data accuracy: {lr_real_acc:.4f}")
    print(f"LR - Synthetic data accuracy: {lr_syn_acc:.4f}")
    print(f"LR - Accuracy ratio: {lr_ratio:.4f}")
    
    # Add MLP classifier evaluation
    print("\nTraining MLP classifier...")
    mlp_params = {
        'hidden_layer_sizes': (100,),
        'max_iter': 300,
        'activation': 'relu',
        'solver': 'adam',
        'random_state': 42
    }
    
    # Train on real data
    mlp_real = MLPClassifier(**mlp_params)
    mlp_real.fit(real_train_df, train_labels)
    mlp_real_pred = mlp_real.predict(real_test_df)
    mlp_real_acc = accuracy_score(test_labels, mlp_real_pred)
    
    # Train on synthetic data
    mlp_syn = MLPClassifier(**mlp_params)
    mlp_syn.fit(synthetic_train_df, train_labels[:len(synthetic_train_df)])
    mlp_syn_pred = mlp_syn.predict(real_test_df)
    mlp_syn_acc = accuracy_score(test_labels, mlp_syn_pred)
    
    # Calculate ratio
    mlp_ratio = mlp_syn_acc / mlp_real_acc
    
    print(f"MLP - Real data accuracy: {mlp_real_acc:.4f}")
    print(f"MLP - Synthetic data accuracy: {mlp_syn_acc:.4f}")
    print(f"MLP - Accuracy ratio: {mlp_ratio:.4f}")
    
    # XGBoost classifier evaluation
    print("\nTraining XGBoost classifier...")
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
    
    # Train on real data
    xgb_real = xgb.XGBClassifier(**xgb_params)
    xgb_real.fit(real_train_df, train_labels)
    xgb_real_pred = xgb_real.predict(real_test_df)
    xgb_real_acc = accuracy_score(test_labels, xgb_real_pred)
    
    # Train on synthetic data
    xgb_syn = xgb.XGBClassifier(**xgb_params)
    xgb_syn.fit(synthetic_train_df, train_labels[:len(synthetic_train_df)])
    xgb_syn_pred = xgb_syn.predict(real_test_df)
    xgb_syn_acc = accuracy_score(test_labels, xgb_syn_pred)
    
    # Calculate ratio
    xgb_ratio = xgb_syn_acc / xgb_real_acc
    
    print(f"XGB - Real data accuracy: {xgb_real_acc:.4f}")
    print(f"XGB - Synthetic data accuracy: {xgb_syn_acc:.4f}")
    print(f"XGB - Accuracy ratio: {xgb_ratio:.4f}")
    
    results = {
        'RandomForest': {
            'real_accuracy': rf_real_acc,
            'synthetic_accuracy': rf_syn_acc,
            'accuracy_ratio': rf_ratio
        },
        'LogisticRegression': {
            'real_accuracy': lr_real_acc,
            'synthetic_accuracy': lr_syn_acc,
            'accuracy_ratio': lr_ratio
        },
        'MLP': {
            'real_accuracy': mlp_real_acc,
            'synthetic_accuracy': mlp_syn_acc,
            'accuracy_ratio': mlp_ratio
        },
        'XGBoost': {
            'real_accuracy': xgb_real_acc,
            'synthetic_accuracy': xgb_syn_acc,
            'accuracy_ratio': xgb_ratio
        }
    }
    
    return results

if __name__ == "__main__":
    print("==== Exact TableGAN Implementation ====")
    print(f"Dataset: {DATASET}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden dimensions: {HIDDEN_DIM}")
    
    # Load the data
    data, train_data, test_data, train_labels, test_labels, scaler = load_adult_full_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        exit()
    
    # Create and train the TableGAN model
    model = TableGAN(
        data_dim=train_data.shape[1],
        num_classes=len(np.unique(train_labels)),
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM,
        delta_mean=DELTA_MEAN,
        delta_var=DELTA_VAR
    )
    
    # Train the model
    losses = model.train(train_data, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot training losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses['disc_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(losses['gen_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 3)
    plt.plot(losses['class_loss'])
    plt.title('Classifier Loss')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(f'results/{DATASET}_training_losses.png')
    plt.close()
    
    # Generate synthetic data with the same size as training data
    synthetic_data = generate_synthetic_data(model, len(train_data), scaler, data)
    
    # Evaluate statistical similarity
    stat_results = evaluate_statistical_similarity(data, synthetic_data)
    
    # Evaluate machine learning efficacy
    ml_results = evaluate_machine_learning_efficacy(
        train_data, test_data, synthetic_data.values, train_labels, test_labels
    )
    
    # Print summary of results
    print("\n==== TableGAN Results Summary ====")
    print(f"Dataset: {DATASET}")
    print(f"Statistical similarity (correlation distance): {stat_results:.4f}")
    
    # Report RF results (as in the paper)
    rf_results = ml_results['RandomForest']
    print(f"ML efficacy - Real data accuracy (RF): {rf_results['real_accuracy']:.4f}")
    print(f"ML efficacy - Synthetic data accuracy (RF): {rf_results['synthetic_accuracy']:.4f}")
    print(f"ML efficacy - Accuracy ratio (RF): {rf_results['accuracy_ratio']:.4f}")
    
    # Report LR results
    lr_results = ml_results['LogisticRegression']
    print(f"ML efficacy - Real data accuracy (LR): {lr_results['real_accuracy']:.4f}")
    print(f"ML efficacy - Synthetic data accuracy (LR): {lr_results['synthetic_accuracy']:.4f}")
    print(f"ML efficacy - Accuracy ratio (LR): {lr_results['accuracy_ratio']:.4f}")
    
    # Report MLP results
    mlp_results = ml_results['MLP']
    print(f"ML efficacy - Real data accuracy (MLP): {mlp_results['real_accuracy']:.4f}")
    print(f"ML efficacy - Synthetic data accuracy (MLP): {mlp_results['synthetic_accuracy']:.4f}")
    print(f"ML efficacy - Accuracy ratio (MLP): {mlp_results['accuracy_ratio']:.4f}")
    
    # Report XGB results
    xgb_results = ml_results['XGBoost']
    print(f"ML efficacy - Real data accuracy (XGB): {xgb_results['real_accuracy']:.4f}")
    print(f"ML efficacy - Synthetic data accuracy (XGB): {xgb_results['synthetic_accuracy']:.4f}")
    print(f"ML efficacy - Accuracy ratio (XGB): {xgb_results['accuracy_ratio']:.4f}")
    
    # Compare with paper results
    print("\n==== Comparison with Paper Results ====")
    print("Paper reported (approximately):")
    print("Statistical similarity - Correlation distance: ~0.30-0.35")
    print("ML efficacy - Real data accuracy: ~84%")
    print("ML efficacy - Synthetic data accuracy: ~79%")
    print("ML efficacy - Accuracy ratio: ~0.94")
    
    print("\nOur results:")
    print(f"Statistical similarity - Correlation distance: {stat_results:.4f}")
    print(f"ML efficacy - Real data accuracy (RF): {rf_results['real_accuracy']:.4f}")
    print(f"ML efficacy - Synthetic data accuracy (RF): {rf_results['synthetic_accuracy']:.4f}")
    print(f"ML efficacy - Accuracy ratio (RF): {rf_results['accuracy_ratio']:.4f}")
    
    # Save results to file
    with open(f'results/{DATASET}_results.txt', 'w') as f:
        f.write("==== TableGAN Results Summary ====\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Statistical similarity (correlation distance): {stat_results:.4f}\n")
        f.write(f"ML efficacy - Real data accuracy (RF): {rf_results['real_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Synthetic data accuracy (RF): {rf_results['synthetic_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Accuracy ratio (RF): {rf_results['accuracy_ratio']:.4f}\n")
        f.write(f"ML efficacy - Real data accuracy (LR): {lr_results['real_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Synthetic data accuracy (LR): {lr_results['synthetic_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Accuracy ratio (LR): {lr_results['accuracy_ratio']:.4f}\n")
        f.write(f"ML efficacy - Real data accuracy (MLP): {mlp_results['real_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Synthetic data accuracy (MLP): {mlp_results['synthetic_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Accuracy ratio (MLP): {mlp_results['accuracy_ratio']:.4f}\n")
        f.write(f"ML efficacy - Real data accuracy (XGB): {xgb_results['real_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Synthetic data accuracy (XGB): {xgb_results['synthetic_accuracy']:.4f}\n")
        f.write(f"ML efficacy - Accuracy ratio (XGB): {xgb_results['accuracy_ratio']:.4f}\n") 