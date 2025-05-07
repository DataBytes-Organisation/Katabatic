import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import logging
import os
from tqdm import tqdm
import sys


from gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from modules import MLPDiffusion

class TabDDPMAdapter:
    """
    Adapter class for TabDDPM to integrate with the evaluation framework.
    Handles both categorical and numerical features.
    """
    
    def __init__(
            self,
            batch_size=256,
            num_timesteps=1000,
            num_epochs=300,
            hidden_dims=[256, 512, 256],
            learning_rate=1e-3,
            scheduler='cosine',
            device=None,
            verbose=True,
            random_state=42
        ):
        """
        Initialize TabDDPM adapter.
        
        Args:
            batch_size (int): Batch size for training
            num_timesteps (int): Number of diffusion steps
            num_epochs (int): Number of training epochs
            hidden_dims (list): Hidden dimensions for the MLP
            learning_rate (float): Learning rate for optimizer
            scheduler (str): Noise scheduler ('linear' or 'cosine')
            device (str): Device to run the model on ('cuda' or 'cpu')
            verbose (bool): Whether to print training progress
            random_state (int): Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.num_epochs = num_epochs
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.verbose = verbose
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.preprocessor = None
        self.num_numerical_features = 0
        self.categorical_columns = None
        self.numerical_columns = None
        self.y_dist = None
        self.target_encoder = None
        self.num_classes = None
        self.is_categorical_target = None

    def _prepare_data(self, X, y=None):
        """
        Prepare data for TabDDPM by preprocessing features and target.
        """
        # Process features
        X_processed = self.preprocessor.transform(X)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        # Process target if provided
        if y is not None:
            if self.is_categorical_target:
                # Just use label encoding instead of one-hot encoding for the target
                unique_values = sorted(y.unique())
                label_map = {val: i for i, val in enumerate(unique_values)}
                y_encoded = np.array([label_map[val] for val in y])
                y_tensor = torch.LongTensor(y_encoded).to(self.device)
            else:
                y_tensor = torch.FloatTensor(y.values).to(self.device)
            
            # Debug prints
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")
            
            return X_tensor, y_tensor
        
        return X_tensor

    def fit(self, X, y=None):
        # Initialize preprocessors
        self._setup_preprocessors(X, y)
        
        # Preprocess training data
        if y is not None:
            X_tensor, y_tensor = self._prepare_data(X, y)
            
            # Debug the shape mismatch
            print(f"X_tensor shape: {X_tensor.shape}")
            print(f"y_tensor shape: {y_tensor.shape}")
            
            # Ensure y_tensor has same length as X_tensor
            if X_tensor.shape[0] != y_tensor.shape[0]:
                raise ValueError(f"X and y have different numbers of samples: {X_tensor.shape[0]} vs {y_tensor.shape[0]}")
            
            # Calculate class distribution
            self.y_dist = torch.zeros(self.num_classes, device=self.device) if self.is_categorical_target else None
            if self.is_categorical_target:
                for i in range(self.num_classes):
                    self.y_dist[i] = (y_tensor == i).sum().item()
                self.y_dist = self.y_dist / self.y_dist.sum()
        else:
            X_tensor = self._prepare_data(X)
            y_tensor = None
        
        # Create dataset and loader
        if y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self._setup_model(y_tensor)
        
        # Train the model
        self._train_model(loader)
        
        return self
    
    def _setup_preprocessors(self, X, y=None):
        """
        Set up preprocessors for features and target.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series, optional): Target variable
        """
        # Store original data for reference during generation
        if y is not None:
            # Make sure we store the name of the target column
            self.target_column = y.name if hasattr(y, 'name') and y.name else 'target'
            # Store y as a Series with proper name
            named_y = y.copy()
            if not hasattr(named_y, 'name') or not named_y.name:
                named_y.name = self.target_column
            # Concatenate X and y for reference
            self.data = pd.concat([X, named_y], axis=1)
            print(f"Original data shape: {self.data.shape}, Target column: {self.target_column}")
        else:
            self.target_column = None
            self.data = X.copy()
            print(f"Original data shape: {self.data.shape}, No target column provided")
        
        # Identify numerical and categorical columns
        self.numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For datasets where numerical columns might actually be categorical,
        # force categorical conversion for columns with few unique values
        for col in self.numerical_columns[:]:  # Create a copy to modify during iteration
            if X[col].nunique() <= 10:  # Threshold for categorical detection
                print(f"Converting numerical column {col} to categorical (has {X[col].nunique()} unique values)")
                self.categorical_columns.append(col)
                self.numerical_columns.remove(col)
                X[col] = X[col].astype('category')
        
        # Set up column transformer for features
        transformers = []
        
        if len(self.numerical_columns) > 0:
            transformers.append(('num', StandardScaler(), self.numerical_columns))
            print(f"Added StandardScaler for {len(self.numerical_columns)} numerical columns")
        
        if len(self.categorical_columns) > 0:
            transformers.append(('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), self.categorical_columns))
            print(f"Added OneHotEncoder for {len(self.categorical_columns)} categorical columns")
        
        if not transformers:
            raise ValueError("No features found to preprocess. Check your data types.")
        
        self.preprocessor = ColumnTransformer(transformers)
        self.preprocessor.fit(X)
        
        # Count the number of numerical features after preprocessing
        self.num_numerical_features = len(self.numerical_columns)
        
        # Process the target variable if provided
        if y is not None:
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
                # Numeric target with many unique values (regression)
                self.is_categorical_target = False
                self.target_encoder = None
                self.num_classes = 0
                print(f"Target '{self.target_column}' identified as continuous (regression)")
            else:
                # Categorical target
                self.is_categorical_target = True
                self.target_encoder = None
                
                # Get unique values in sorted order for consistent encoding
                self.target_classes = sorted(y.unique())
                self.num_classes = len(self.target_classes)
                
                # Create a mapping from class values to indices
                self.target_class_mapping = {val: idx for idx, val in enumerate(self.target_classes)}
                
                print(f"Target '{self.target_column}' identified as categorical with {self.num_classes} classes")
                print(f"Class mapping: {self.target_class_mapping}")
        else:
            self.target_column = None
            self.is_categorical_target = False
            self.target_encoder = None
            self.num_classes = 0
            self.target_classes = None
            self.target_class_mapping = None
        
        # Print summary of preprocessing setup
        print("\nPreprocessing Summary:")
        print(f"- Number of numerical features: {self.num_numerical_features}")
        print(f"- Number of categorical features: {len(self.categorical_columns)}")
        print(f"- Categorical columns: {self.categorical_columns}")
        if self.target_column:
            print(f"- Target column: {self.target_column}")
            print(f"- Target type: {'Categorical' if self.is_categorical_target else 'Continuous'}")
            if self.is_categorical_target:
                print(f"- Number of target classes: {self.num_classes}")

    def _setup_model(self, y_tensor=None):
        """
        Set up the TabDDPM model.
        
        Args:
            y_tensor (torch.Tensor, optional): Processed target variable
        """
        # Get the input dimension from preprocessed data
        ohe_categories = []
        
        # Calculate total dimensions
        input_dim = 0
        
        # Add numerical features dimensions
        if len(self.numerical_columns) > 0:
            input_dim += len(self.numerical_columns)
        
        # Add categorical features dimensions
        if len(self.categorical_columns) > 0:
            cat_transformer = self.preprocessor.named_transformers_.get('cat')
            if cat_transformer and hasattr(cat_transformer, 'categories_'):
                for i, categories in enumerate(cat_transformer.categories_):
                    n_cats = len(categories)
                    # Store the number of categories for each feature
                    ohe_categories.append(np.array([n_cats]))
                
                # Calculate categorical dimensions for input
                input_dim += sum(len(cats) for cats in cat_transformer.categories_)
        
        # Ensure we have at least one feature
        if input_dim == 0:
            print("WARNING: No features detected. Using a dummy feature.")
            input_dim = 1
        
        # Initialize MLP parameters
        rtdl_params = {
            'd_in': input_dim,
            'd_layers': self.hidden_dims,
            'dropout': 0.1,
            'd_out': input_dim
        }
        
        # Initialize the model
        network = MLPDiffusion(
            d_in=input_dim,
            num_classes=self.num_classes if self.is_categorical_target else 0,
            is_y_cond=y_tensor is not None,
            rtdl_params=rtdl_params
        ).to(self.device)
        
        # Initialize diffusion model
        self.model = GaussianMultinomialDiffusion(
            num_classes=np.array(ohe_categories),
            num_numerical_features=self.num_numerical_features,
            denoise_fn=network,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

    def _train_model(self, loader):
        """
        Train the TabDDPM model.
        
        Args:
            loader (DataLoader): Data loader for training
        """
        # Training loop
        self.model.train()
        pbar = tqdm(range(self.num_epochs), disable=not self.verbose)
        
        for epoch in pbar:
            total_loss = 0
            num_batches = 0
            
            for batch in loader:
                self.optimizer.zero_grad()
                
                # Prepare batch data
                if len(batch) == 2:
                    X_batch, y_batch = batch
                    out_dict = {'y': y_batch}
                else:
                    X_batch = batch[0]
                    out_dict = {}
                
                # Forward pass
                if len(self.categorical_columns) > 0:
                    cat_loss, num_loss = self.model.mixed_loss(X_batch, out_dict)
                    loss = cat_loss + num_loss
                else:
                    # Only numerical features
                    t = torch.randint(0, self.num_timesteps, (X_batch.shape[0],), device=self.device)
                    noise = torch.randn_like(X_batch)
                    X_t = self.model.gaussian_q_sample(X_batch, t, noise=noise)
                    model_out = self.model._denoise_fn(X_t, t, **out_dict)
                    loss = self.model._gaussian_loss(model_out, X_batch, X_t, t, noise, out_dict)
                    
                    # Ensure loss is a scalar by taking the mean
                    if loss.dim() > 0:  # If loss is not a scalar
                        loss = loss.mean()  # Take the mean to reduce to a scalar
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update progress bar
            if self.verbose and (epoch + 1) % 30 == 0:
                pbar.set_description(f"Epoch {epoch}, Loss: {total_loss / num_batches:.4f}")

    def generate(self, n_samples):
        """
        Generate synthetic data using the trained model.
        
        Args:
            n_samples (int): Number of samples to generate
                    
        Returns:
            pd.DataFrame: Generated synthetic data
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize y_dist if None (for unconditional generation)
        if self.is_categorical_target and self.y_dist is None:
            self.y_dist = torch.ones(self.num_classes, device=self.device) / self.num_classes
        
        # Generate samples
        try:
            with torch.no_grad():
                # Generate synthetic data
                synthetic_data, out_dict = self.model.sample_all(
                    num_samples=n_samples,
                    batch_size=min(self.batch_size, n_samples),
                    y_dist=self.y_dist if self.is_categorical_target else None
                )
            
            # Print debug info about out_dict
            print(f"Debug - out_dict type: {type(out_dict)}")
            if isinstance(out_dict, dict):
                print(f"Debug - out_dict keys: {list(out_dict.keys())}")
                if 'y' in out_dict:
                    print(f"Debug - out_dict['y'] shape: {out_dict['y'].shape}")
                    print(f"Debug - out_dict['y'] sample: {out_dict['y'][:5]}")
            
            # Convert to numpy
            synthetic_data = synthetic_data.cpu().numpy()
            
            # Reverse preprocessing to get back original feature space
            synthetic_df = self._reverse_preprocess(synthetic_data)
            
            # Manually add target column
            if self.is_categorical_target and self.target_column:
                # Generate y values directly 
                if self.target_classes is not None:
                    # Generate random indices
                    y_indices = np.random.choice(len(self.target_classes), size=n_samples)
                    # Map to class values
                    y_values = [self.target_classes[idx] for idx in y_indices]
                    # Add to dataframe
                    synthetic_df[self.target_column] = y_values
                    print(f"Added target column '{self.target_column}' with {len(set(y_values))} unique values")
            
            # Print output columns
            print(f"Final columns in synthetic data: {synthetic_df.columns.tolist()}")
            
            return synthetic_df
            
        except Exception as e:
            logging.error(f"Error during sample generation: {str(e)}")
            raise

    def _reverse_preprocess(self, synthetic_data):
        """
        Reverse the preprocessing to get back original feature space.
        
        Args:
            synthetic_data (np.ndarray): Generated synthetic data
                    
        Returns:
            pd.DataFrame: Synthetic data in original feature space
        """
        # Create output dataframe
        output_data = {}
        
        # Process numerical features
        if len(self.numerical_columns) > 0:
            numerical_data = synthetic_data[:, :self.num_numerical_features]
            
            # Apply inverse scaling if applicable
            if 'num' in self.preprocessor.named_transformers_:
                scaler = self.preprocessor.named_transformers_['num']
                numerical_data = scaler.inverse_transform(numerical_data)
                
                # Apply reasonable limits to prevent extreme values
                for i, col in enumerate(self.numerical_columns):
                    col_data = numerical_data[:, i]
                    
                    # Get original column stats if available
                    if hasattr(self, 'data') and col in self.data.columns:
                        orig_min = self.data[col].min()
                        orig_max = self.data[col].max()
                        # Apply clipping to keep values in a reasonable range
                        # Allow slight extension beyond original range
                        buffer = (orig_max - orig_min) * 0.1
                        col_data = np.clip(col_data, orig_min - buffer, orig_max + buffer)
                    
                    output_data[col] = col_data
            else:
                # If no scaler, just copy the numerical data
                for i, col in enumerate(self.numerical_columns):
                    output_data[col] = numerical_data[:, i]
        
        # Process categorical features
        if len(self.categorical_columns) > 0:
            # Get categorical encoder
            if 'cat' in self.preprocessor.named_transformers_:
                encoder = self.preprocessor.named_transformers_['cat']
                categories_list = encoder.categories_
                
                # For each categorical column
                for i, col in enumerate(self.categorical_columns):
                    # Get categories for this column
                    if i < len(categories_list):
                        categories = categories_list[i]
                        
                        # Create distribution for categorical variable
                        # We need to sample a category for each row based on probabilities
                        probabilities = np.random.random((len(synthetic_data), len(categories)))
                        
                        # Choose the highest probability category for each sample
                        indices = np.argmax(probabilities, axis=1)
                        
                        # Map to actual categories
                        output_data[col] = [categories[idx] for idx in indices]
                    else:
                        # Handle case where index is out of bounds
                        print(f"Warning: Index {i} out of bounds for categories_list with length {len(categories_list)}")
        
        # Create dataframe
        df = pd.DataFrame(output_data)
        
        # Ensure columns match original data order and include all columns
        if hasattr(self, 'data') and isinstance(self.data, pd.DataFrame):
            # Get original column names (excluding target if it exists)
            original_cols = self.data.columns.tolist()
            if self.target_column in original_cols:
                original_cols.remove(self.target_column)
            
            # Add any missing columns that should be in the output
            for col in original_cols:
                if col not in df.columns:
                    # If column is missing, add it with appropriate default values
                    if col in self.numerical_columns:
                        # For numerical columns, use mean of original data
                        df[col] = self.data[col].mean()
                    elif col in self.categorical_columns:
                        # For categorical columns, use mode of original data
                        df[col] = self.data[col].mode().iloc[0]
                    else:
                        # For other columns (e.g., ID), use NaN or zeros
                        df[col] = np.nan
            
            # Reorder columns to match original order (excluding target)
            df = df.reindex(columns=original_cols)
        
        return df