import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from katabatic.katabatic_spi import KatabaticModelSPI


class DataTransformer:
    def __init__(self, max_clusters=10):
        """
        Initialize the DataTransformer.

        Args:
            max_clusters (int or str): Maximum number of clusters for GMM.
                                        If 'auto', use the number of unique values.
        """
        if isinstance(max_clusters, str) and max_clusters.lower() == "auto":
            self.max_clusters = None  # Enable auto-detection of clusters
        else:
            self.max_clusters = max_clusters

        # Dictionaries to store encoders, scalers, and Gaussian Mixture Models
        self.encoders = {}
        self.continuous_gmms = {}
        self.output_info = []
        self.output_dim = 0
        self.scalers = {}
        self.dataframe_columns = []

    def fit(self, data):
        """
        Fit the transformer on the provided data.

        Args:
            data (pd.DataFrame): The input data.
        """
        self.output_info = []
        self.output_dim = 0
        self.dataframe_columns = data.columns

        for column in data.columns:
            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                # Process categorical columns
                try:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                except TypeError:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')    
                encoder.fit(data[[column]])
                self.encoders[column] = encoder
                categories = encoder.categories_[0]
                self.output_info.append(('categorical', len(categories)))
                self.output_dim += len(categories)
            else:
                # Process continuous columns
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[[column]])
                self.scalers[column] = scaler

                unique_values = len(np.unique(data[column]))
                if self.max_clusters is None:
                    n_components = unique_values
                else:
                    n_components = min(self.max_clusters, unique_values)

                n_components = max(n_components, 1)  # Ensure at least one component

                # Fit a Bayesian Gaussian Mixture Model
                vgm = BayesianGaussianMixture(
                    n_components=n_components,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    n_init=1,
                    max_iter=1000,
                    random_state=42
                )
                vgm.fit(scaled_data)
                self.continuous_gmms[column] = vgm
                active_components = np.sum(vgm.weights_ > 1e-3)
                self.output_info.append(('continuous', active_components))
                self.output_dim += active_components + 1  # +1 for normalized value

    def transform(self, data):
        """
        Transform the data into a numerical format suitable for the CTGAN model.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            np.ndarray: Transformed data.
        """
        outputs = []
        for idx, column in enumerate(data.columns):
            column_type, info = self.output_info[idx]
            if column_type == 'categorical':
                # Encode categorical columns
                encoder = self.encoders[column]
                transformed = encoder.transform(data[[column]])
                outputs.append(transformed)
            else:
                # Encode continuous columns
                scaler = self.scalers[column]
                x = scaler.transform(data[[column]])
                vgm = self.continuous_gmms[column]

                # Predict probabilities for each Gaussian component
                probs = vgm.predict_proba(x)
                # Sample components based on probabilities
                components = np.array([
                    np.random.choice(len(p), p=p) if p.sum() > 0 else np.random.randint(len(p))
                    for p in probs
                ])

                # Retrieve means and standard deviations of the components
                means = vgm.means_.flatten()
                stds = np.sqrt(vgm.covariances_).flatten()

                # Select means and stds based on sampled components
                selected_means = means[components]
                selected_stds = stds[components]
                selected_stds[selected_stds == 0] = 1e-6  # Avoid division by zero

                # Normalize the continuous values
                normalized_values = ((x.flatten() - selected_means) / (4 * selected_stds)).reshape(-1, 1)
                normalized_values = np.clip(normalized_values, -0.99, 0.99)

                n_components = len(means)
                # Create one-hot encoding for the selected components
                component_one_hot = np.zeros((x.shape[0], n_components))
                component_one_hot[np.arange(x.shape[0]), components] = 1

                # Filter out inactive components
                active_components = vgm.weights_ > 1e-3
                component_one_hot = component_one_hot[:, active_components]

                # Concatenate component encoding with normalized values
                transformed = np.concatenate([component_one_hot, normalized_values], axis=1)
                outputs.append(transformed)

        # Combine all transformed columns
        data_concat = np.concatenate(outputs, axis=1)
        return data_concat.astype('float32')

    def inverse_transform(self, data):
        """
        Inverse transform the data back to its original format.

        Args:
            data (np.ndarray): Transformed data.

        Returns:
            pd.DataFrame: Data in original format.
        """
        recovered_data = {}
        col_idx = 0
        for idx, column in enumerate(self.dataframe_columns):
            column_type, info = self.output_info[idx]
            if column_type == 'categorical':
                # Decode categorical columns
                dim = info
                values = data[:, col_idx:col_idx + dim]
                values = np.argmax(values, axis=1)
                categories = self.encoders[column].categories_[0]
                recovered = categories[values]
                recovered_data[column] = recovered
                col_idx += dim
            else:
                # Decode continuous columns
                n_components = info
                component_probs = data[:, col_idx:col_idx + n_components]
                scalar_values = data[:, col_idx + n_components]
                components = np.argmax(component_probs, axis=1)
                vgm = self.continuous_gmms[column]
                means = vgm.means_.flatten()
                stds = np.sqrt(vgm.covariances_).flatten()
                active_components = vgm.weights_ > 1e-3
                means = means[active_components]
                stds = stds[active_components]
                selected_means = means[components]
                selected_stds = stds[components]
                selected_stds[selected_stds == 0] = 1e-6  # Avoid division by zero

                # Recover the original continuous values
                recovered = scalar_values * 4 * selected_stds + selected_means
                recovered = np.clip(recovered, 0, 1)
                scaler = self.scalers[column]
                recovered = scaler.inverse_transform(recovered.reshape(-1, 1)).flatten()
                recovered_data[column] = recovered
                col_idx += n_components + 1

        df = pd.DataFrame(recovered_data)
        # Ensure continuous columns are correctly typed
        for column in df.columns:
            if column in self.continuous_gmms:
                df[column] = pd.to_numeric(df[column], errors='coerce')
        return df


class DataSampler:
    def __init__(self, data, transformer):
        """
        Initialize the DataSampler.

        Args:
            data (pd.DataFrame): The input data.
            transformer (DataTransformer): The fitted DataTransformer.
        """
        self.transformer = transformer
        self.n = len(data)
        self.data = data
        self.discrete_columns = []
        self.discrete_column_category_counts = []
        self.discrete_column_probs = []
        self.discrete_column_category_values = []

        for idx, column in enumerate(data.columns):
            column_type, info = transformer.output_info[idx]
            if column_type == 'categorical':
                self.discrete_columns.append(column)
                counts = data[column].value_counts()
                probs = counts / counts.sum()
                self.discrete_column_probs.append(probs.values)
                self.discrete_column_category_counts.append(len(counts))
                self.discrete_column_category_values.append(counts.index.values)

        # Normalize probabilities to handle imbalanced datasets
        self.discrete_column_probs = [probs / probs.sum() for probs in self.discrete_column_probs]

    def sample_condvec(self, batch_size):
        """
        Sample conditional vectors for the generator.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            tuple: (cond, mask)
        """
        if not self.discrete_columns:
            return None, None

        # Initialize condition vectors and mask
        cond = np.zeros((batch_size, sum(self.discrete_column_category_counts)), dtype='float32')
        mask = np.ones((batch_size, len(self.discrete_columns)), dtype='int32')

        for i in range(batch_size):
            for idx in range(len(self.discrete_columns)):
                column = self.discrete_columns[idx]
                probs = self.discrete_column_probs[idx]
                categories = self.discrete_column_category_values[idx]
                # Sample a category based on probabilities
                category = np.random.choice(categories, p=probs)
                encoder = self.transformer.encoders[column]
                one_hot = encoder.transform([[category]]).flatten()
                # Assign the one-hot encoded category to the condition vector
                cond[i, self.get_condvec_indices(idx)] = one_hot

        return cond, mask

    def sample_original_condvec(self, batch_size):
        """
        Sample original conditional vectors for data generation.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            np.ndarray: Conditional vectors.
        """
        if not self.discrete_columns:
            return None

        cond = np.zeros((batch_size, sum(self.discrete_column_category_counts)), dtype='float32')

        for i in range(batch_size):
            for idx in range(len(self.discrete_columns)):
                column = self.discrete_columns[idx]
                probs = self.discrete_column_probs[idx]
                categories = self.discrete_column_category_values[idx]
                # Sample a category based on probabilities
                category = np.random.choice(categories, p=probs)
                encoder = self.transformer.encoders[column]
                one_hot = encoder.transform([[category]]).flatten()
                # Assign the one-hot encoded category to the condition vector
                cond[i, self.get_condvec_indices(idx)] = one_hot

        return cond

    def get_condvec_indices(self, idx):
        """
        Get the indices for the conditional vector corresponding to a specific column.

        Args:
            idx (int): Column index.

        Returns:
            np.ndarray: Indices for the conditional vector.
        """
        start = sum(self.discrete_column_category_counts[:idx])
        end = start + self.discrete_column_category_counts[idx]
        return np.arange(start, end)

    def dim_cond_vec(self):
        """
        Get the dimensionality of the conditional vector.

        Returns:
            int: Dimensionality of the conditional vector.
        """
        return sum(self.discrete_column_category_counts)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a Residual Block.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
        """
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        Forward pass through the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the residual connection.
        """
        out = F.leaky_relu(self.bn(self.fc(x)), 0.2)
        return x + out  # Add residual connection


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim):
        """
        Initialize the Generator network.

        Args:
            noise_dim (int): Dimension of the noise vector.
            cond_dim (int): Dimension of the conditional vector.
            output_dim (int): Dimension of the output data.
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.input_dim = noise_dim + cond_dim

        # Define network layers
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.res_block1 = ResidualBlock(256, 256)
        self.res_block2 = ResidualBlock(256, 256)

        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, noise, cond):
        """
        Forward pass through the Generator.

        Args:
            noise (torch.Tensor): Noise vector.
            cond (torch.Tensor): Conditional vector.

        Returns:
            torch.Tensor: Generated data.
        """
        # Concatenate noise and condition
        x = torch.cat([noise, cond], dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim, num_classes):
        """
        Initialize the Discriminator network.

        Args:
            input_dim (int): Dimension of the input data.
            cond_dim (int): Dimension of the conditional vector.
            num_classes (int): Number of classes for auxiliary classification.
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim + cond_dim

        # Define network layers
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)

        # Output layers for adversarial and auxiliary classification
        self.fc_adv = nn.Linear(128, 1)
        self.fc_aux = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (Validity score, Class logits)
        """
        x = x.view(-1, self.input_dim)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout3(x)

        # Output for real/fake classification
        validity = self.fc_adv(x)
        # Output for auxiliary class prediction
        class_logits = self.fc_aux(x)

        return validity, class_logits


class CTGANLoss:
    @staticmethod
    def calc_gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp):
        """
        Calculate the gradient penalty for WGAN-GP.

        Args:
            discriminator (nn.Module): The discriminator model.
            real_data (torch.Tensor): Real data samples.
            fake_data (torch.Tensor): Fake data samples generated by the generator.
            device (torch.device): The device to perform computations on.
            lambda_gp (float): Gradient penalty coefficient.

        Returns:
            torch.Tensor: The gradient penalty.
        """
        batch_size = real_data.size(0)
        # Sample random interpolation coefficients
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)
        # Create interpolated samples
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)

        # Get discriminator output for interpolated samples
        validity_interpolates, _ = discriminator(interpolates)
        # Compute gradients with respect to interpolated samples
        gradients = torch.autograd.grad(
            outputs=validity_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(validity_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Reshape gradients and compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return gradient_penalty


class CtganAdapter(KatabaticModelSPI):
    def __init__(self, **kwargs):
        """
        Initialize the CTGAN Adapter.

        Args:
            **kwargs: Keyword arguments for configuration.
        """
        super().__init__("mixed")
        self.embedding_dim = kwargs.get('noise_dim', 128)
        self.generator_lr = kwargs.get('learning_rate', 1e-4)
        self.discriminator_lr = kwargs.get('learning_rate', 1e-4)
        self.max_batch_size = kwargs.get('batch_size', 500)
        self.batch_size = self.max_batch_size
        self.discriminator_steps = kwargs.get('discriminator_steps', 5)
        self.epochs = kwargs.get('epochs', 300)
        self.lambda_gp = kwargs.get('lambda_gp', 10)
        self.vgm_components = kwargs.get('vgm_components', "auto")  # Enable auto-detection of GMM components
        self.device = torch.device("cuda:0" if kwargs.get('cuda', True) and torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()
        self.discrete_columns = []

    def load_data(self, data):
        """
        Load and preprocess the data.

        Args:
            data (pd.DataFrame): The input data.
        """
        self.data = data
        self.transformer = DataTransformer(max_clusters=self.vgm_components)
        self.transformer.fit(data)
        self.data_sampler = DataSampler(data, self.transformer)

        self.output_dim = self.transformer.output_dim
        self.cond_dim = self.data_sampler.dim_cond_vec()
        self.discrete_columns = [
            col for col, col_info in zip(data.columns, self.transformer.output_info)
            if col_info[0] == 'categorical'
        ]

        # Determine the target variable (assumed to be the last categorical column)
        if self.discrete_columns:
            self.target_column = self.discrete_columns[-1]
            self.num_classes = self.transformer.encoders[self.target_column].categories_[0].shape[0]
        else:
            self.target_column = None
            self.num_classes = 0

    def load_model(self):
        """
        Initialize the generator and discriminator models along with their optimizers and schedulers.
        """
        self.generator = Generator(
            noise_dim=self.embedding_dim,
            cond_dim=self.cond_dim,
            output_dim=self.output_dim
        ).to(self.device)

        self.discriminator = Discriminator(
            input_dim=self.output_dim,
            cond_dim=self.cond_dim,
            num_classes=self.num_classes
        ).to(self.device)

        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.9)
        )

        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.9)
        )

        self.schedulerG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=100, gamma=0.5)
        self.schedulerD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=100, gamma=0.5)

    def fit(self, X, y=None):
        """
        Fit the CTGAN model to the data.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series, optional): Target labels.
        """
        if y is not None:
            # Combine features and target into a single DataFrame
            data = pd.concat([X.reset_index(drop=True), pd.Series(y, name='Category').reset_index(drop=True)], axis=1)
        else:
            data = X
        self.load_data(data)

        # Adjust batch size if dataset is smaller than the maximum batch size
        self.batch_size = min(self.max_batch_size, len(data))
        if self.batch_size < self.max_batch_size:
            print(f"Adjusted batch size to {self.batch_size} due to small dataset size")

        self.load_model()
        self.train()

    def train(self):
        """
        Train the generator and discriminator models.
        """
        # Transform the data using the fitted transformer
        data = self.transformer.transform(self.data)
        dataset = TensorDataset(torch.FloatTensor(data))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        log_interval = max(1, self.epochs // 10)

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            epoch_loss_d = 0
            epoch_loss_g = 0
            n_batches = 0

            for id_, data_batch in enumerate(data_loader):
                real = data_batch[0].to(self.device)
                current_batch_size = real.size(0)

                # Sample conditional vectors
                condvec, mask = self.data_sampler.sample_condvec(current_batch_size)
                if condvec is not None:
                    c1 = torch.from_numpy(condvec).to(self.device)
                else:
                    c1 = torch.zeros(current_batch_size, self.cond_dim, device=self.device)

                # Train Discriminator multiple times per generator step
                for _ in range(self.discriminator_steps):
                    noise = torch.randn(current_batch_size, self.embedding_dim, device=self.device)
                    fake = self.generator(noise, c1)

                    # Concatenate fake data with conditions and detach to prevent gradients flowing to generator
                    fake_cat = torch.cat([fake, c1], dim=1).detach()
                    real_cat = torch.cat([real, c1], dim=1)

                    # Get discriminator outputs
                    validity_real, class_logits_real = self.discriminator(real_cat)
                    validity_fake, _ = self.discriminator(fake_cat)

                    # Compute Wasserstein adversarial loss
                    loss_adv = -torch.mean(validity_real) + torch.mean(validity_fake)

                    # Compute gradient penalty
                    gp = CTGANLoss.calc_gradient_penalty(self.discriminator, real_cat, fake_cat, self.device, self.lambda_gp)

                    # Compute classification loss if applicable
                    if self.num_classes > 0:
                        target_labels = torch.argmax(c1[:, -self.num_classes:], dim=1)
                        class_loss = F.cross_entropy(class_logits_real, target_labels)
                        loss_d = loss_adv + gp + class_loss
                    else:
                        loss_d = loss_adv + gp

                    # Backpropagate and update discriminator
                    self.optimizerD.zero_grad()
                    loss_d.backward()
                    self.optimizerD.step()

                    epoch_loss_d += loss_d.item()

                # Train Generator
                noise = torch.randn(current_batch_size, self.embedding_dim, device=self.device)
                fake = self.generator(noise, c1)
                fake_cat = torch.cat([fake, c1], dim=1)

                # Get discriminator output for fake data
                validity_fake, class_logits_fake = self.discriminator(fake_cat)

                # Compute generator adversarial loss
                loss_g_adv = -torch.mean(validity_fake)

                # Compute classification loss if applicable
                if self.num_classes > 0:
                    target_labels = torch.argmax(c1[:, -self.num_classes:], dim=1)
                    class_loss_g = F.cross_entropy(class_logits_fake, target_labels)
                    loss_g = loss_g_adv + class_loss_g
                else:
                    loss_g = loss_g_adv

                # Backpropagate and update generator
                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

                epoch_loss_g += loss_g.item()
                n_batches += 1

            # Update learning rates
            self.schedulerG.step()
            self.schedulerD.step()

            # Logging at specified intervals
            if epoch % log_interval == 0:
                if n_batches > 0:
                    print(f"Epoch {epoch}, Loss D: {epoch_loss_d / n_batches:.4f}, Loss G: {epoch_loss_g / n_batches:.4f}")
                else:
                    print(f"Epoch {epoch}, No batches processed")

        self.writer.close()

    def generate(self, n):
        """
        Generate synthetic data.

        Args:
            n (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Generated synthetic data.
        """
        try:
            self.generator.eval()
            data = []
            steps = n // self.batch_size + 1
            for _ in range(steps):
                noise = torch.randn(self.batch_size, self.embedding_dim, device=self.device)
                condvec = self.data_sampler.sample_original_condvec(self.batch_size)
                if condvec is not None:
                    c1 = torch.from_numpy(condvec).to(self.device)
                else:
                    c1 = torch.zeros(self.batch_size, self.cond_dim, device=self.device)
                with torch.no_grad():
                    fake = self.generator(noise, c1)
                data.append(fake.cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]  # Trim to desired number of samples
            data = self.transformer.inverse_transform(data)
            return data
        except Exception as e:
            print(f"An error occurred during data generation: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of failure


# Example usage with updated configuration
if __name__ == "__main__":
    import json

    # Define configuration parameters
    config = {
        "ctgan_params": {
            "noise_dim": 128,
            "learning_rate": 2e-4,
            "batch_size": 500,
            "discriminator_steps": 5,
            "epochs": 300,
            "lambda_gp": 10,
            "pac": 10,
            "cuda": True,
            "vgm_components": "auto"  # Enable auto-detection of GMM components
        },
        "evaluation": {
            "test_size": 0.2,
            "random_state": 42
        },
        "visualization": {
            "n_features": 5,
            "figsize": [15, 20]
        }
    }

    # Example datasets (replace with actual data loading as needed)
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits

    datasets = {
        "iris": load_iris(),
        "breast_cancer": load_breast_cancer(),
        "wine": load_wine(),
        "digits": load_digits()
    }

    for name, dataset in datasets.items():
        print(f"Processing {name} dataset")
        try:
            if name == "digits":
                # Digits dataset is already numerical
                data = pd.DataFrame(
                    dataset.data,
                    columns=dataset.feature_names if hasattr(dataset, 'feature_names') else [f"pixel_{i}" for i in range(dataset.data.shape[1])]
                )
                data['target'] = dataset.target.astype(str)  # Convert target to string for categorical handling
            else:
                data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
                if 'target' in dataset:
                    data['target'] = dataset.target.astype(str)
            ctgan = CtganAdapter(**config["ctgan_params"])
            ctgan.fit(data.drop('target', axis=1), y=data['target'])
            synthetic_data = ctgan.generate(n=len(data))
            print(f"Synthetic data for {name} generated successfully.\n")
        except Exception as e:
            print(f"Error fitting CTGAN model for {name}: {e}\n")
            print(f"Error processing {name} dataset: {e}\n")
            print(f"Failed to generate synthetic data for {name}\n")

    print("Experiment completed.")
