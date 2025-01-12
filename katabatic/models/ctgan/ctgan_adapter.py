from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import torch as torch
from .ctgan import CTGAN

class CtganAdapter(KatabaticModelSPI):
    def __init__(self,
                 embedding_dim=128,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=1,
                 log_frequency=True,
                 verbose=False,
                 epochs=300,
                 pac=10,
                 cuda=True,
                 discrete_columns=None):
        """
        Initialize the CtganAdapter with specified hyperparameters.
        """
        assert batch_size % 2 == 0, "Batch size must be even."
        
        self.discrete_columns = discrete_columns
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        # Check for CUDA availability
        if not cuda or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        # Initialize the CTGAN model
        self.model = CTGAN(
            embedding_dim=self._embedding_dim,
            generator_dim=self._generator_dim,
            discriminator_dim=self._discriminator_dim,
            generator_lr=self._generator_lr,
            generator_decay=self._generator_decay,
            discriminator_lr=self._discriminator_lr,
            discriminator_decay=self._discriminator_decay,
            batch_size=self._batch_size,
            discriminator_steps=self._discriminator_steps,
            log_frequency=self._log_frequency,
            verbose=self._verbose,
            epochs=self._epochs,
            pac=self.pac,
            device=self.device
        )
