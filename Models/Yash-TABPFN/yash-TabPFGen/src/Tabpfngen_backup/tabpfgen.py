import numpy as np
import torch
import torch.nn.functional as F
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._encoders')

class TabPFGen:
    def __init__(
        self,
        n_sgld_steps: int = 1000,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto"
    ):
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.scaler = StandardScaler()
        self.device = self._infer_device(device)

    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        if (device is None) or (isinstance(device, str) and device == "auto"):
            device_type_ = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_type_)
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")
        
    def _compute_energy(self, x_synth, y_synth, x_train, y_train) -> torch.Tensor:
        distances = torch.cdist(x_synth, x_train)
        min_distances, _ = distances.min(dim=1)
        class_mask = (y_synth.unsqueeze(1) == y_train.unsqueeze(0))
        class_distances = distances * class_mask.float()
        class_distances = class_distances.sum(dim=1) / (class_mask.float().sum(dim=1) + 1e-6)
        energy = min_distances + class_distances
        return energy

    def _sgld_step(self, x_synth, y_synth, x_train, y_train) -> torch.Tensor:
        x_synth = x_synth.clone().detach().requires_grad_(True)
        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        energy_sum = energy.sum()
        grad = torch.autograd.grad(energy_sum, x_synth, create_graph=False, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_synth)
        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)
        x_synth_new = x_synth - self.sgld_step_size * grad + self.sgld_noise_scale * noise
        return x_synth_new

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_samples: int,
        balance_classes: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.fit_transform(X_train)
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device)

        if balance_classes:
            classes = np.unique(y_train.cpu().numpy())
            n_per_class = num_samples // len(classes)
            x_synth_list = []
            y_synth_list = []
            for cls in classes:
                idx = np.where(y_train.cpu().numpy() == cls)[0]
                sample_idx = np.random.choice(idx, size=n_per_class)
                x_init = x_train[sample_idx] + torch.randn(n_per_class, X_train.shape[1], device=self.device) * 0.01
                y_init = torch.full((n_per_class,), cls, device=self.device)
                x_synth_list.append(x_init)
                y_synth_list.append(y_init)
            x_synth = torch.cat(x_synth_list, dim=0)
            y_synth = torch.cat(y_synth_list, dim=0)
        else:
            x_synth = torch.randn(num_samples, X_train.shape[1], device=self.device) * 0.01
            y_synth = torch.randint(0, len(np.unique(y_train)), (num_samples,), device=self.device)

        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train)
            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()
        x_train_np = x_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        clf = TabPFNClassifier(device=self.device)
        clf.fit(x_train_np, y_train_np)
        probs = clf.predict_proba(x_synth_np)
        y_synth = torch.tensor(probs.argmax(axis=1), device=self.device)
        X_synth = self.scaler.inverse_transform(x_synth.detach().cpu().numpy())
        y_synth = y_synth.cpu().numpy()

        return X_synth, y_synth

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_samples: int,
        use_quantiles: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        regressor = TabPFNRegressor(device=self.device)
        X_scaled = self.scaler.fit_transform(X_train)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_scaled = (y_train - y_mean) / y_std
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        n_strata = 10
        y_strata = np.quantile(y_train, np.linspace(0, 1, n_strata+1))
        x_synth_list = []
        samples_per_stratum = num_samples // n_strata
        for i in range(n_strata):
            mask = (y_train >= y_strata[i]) & (y_train <= y_strata[i+1])
            stratum_indices = np.where(mask)[0]
            if len(stratum_indices) > 0:
                sampled_indices = np.random.choice(stratum_indices, size=samples_per_stratum)
                x_stratum = X_scaled[sampled_indices]
                stratum_std = np.std(x_stratum, axis=0)
                noise = np.random.normal(0, stratum_std * 0.1, (samples_per_stratum, X_train.shape[1]))
                x_synth_list.append(x_stratum + noise)

        x_synth_init = np.vstack(x_synth_list)
        x_synth = torch.tensor(x_synth_init, device=self.device, dtype=torch.float32)

        adaptive_step_size = self.sgld_step_size
        for step in range(self.n_sgld_steps):
            if step % 100 == 0:
                adaptive_step_size *= 0.9
            x_synth = self._sgld_step(
                x_synth,
                torch.zeros(len(x_synth), device=self.device),
                x_train,
                torch.zeros_like(torch.tensor(y_scaled, device=self.device))
            )
            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()
        try:
            regressor.fit(X_scaled, y_scaled)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = regressor.predict(x_synth_np, output_type='full')

            if use_quantiles:
                quantiles = predictions['quantiles']
                if not isinstance(quantiles, list):
                    quantiles = [quantiles]
                n_quantiles = len(quantiles)
                probs = np.abs(np.random.normal(0, 0.5, size=len(x_synth_np)))
                quantile_idx = (probs * n_quantiles).astype(int).clip(0, n_quantiles-1)
                y_synth = np.array([quantiles[i][j] for j, i in enumerate(quantile_idx)])
            else:
                y_synth = np.array(predictions['median'])
            y_synth += np.random.normal(0, 0.01, size=len(y_synth))

        except Exception as e:
            print(f"Warning: Error in regression prediction: {str(e)}")
            print("Falling back to stratified sampling...")
            y_synth = np.random.normal(y_mean, y_std, size=len(x_synth_np))

        X_synth = self.scaler.inverse_transform(x_synth_np)
        y_synth = y_synth * y_std + y_mean

        return X_synth, y_synth
