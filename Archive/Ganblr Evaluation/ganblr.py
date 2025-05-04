import torch
import torch.nn as nn
import pandas as pd
import logging  # Import logging module

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GANBLR:
    def __init__(self, input_dim):
        self.generator = self.build_generator(output_dim=input_dim)
        self.discriminator = self.build_discriminator(input_dim=input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        logging.info("GANBLR Model initialized.")

    def build_generator(self, output_dim):
        return nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def build_discriminator(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def fit(self, data):
        try:
            data_tensor = torch.tensor(data.values, dtype=torch.float32)
            logging.info("Training the GANBLR model...")
        except Exception as e:
            logging.error(f"Error while preparing data: {e}")
            raise
        
        for epoch in range(100):
            try:
                noise = torch.randn(data_tensor.size(0), 100)
                self.optimizer_G.zero_grad()
                generated_data = self.generator(noise)
                validity = self.discriminator(generated_data)
                g_loss = self.criterion(validity, torch.ones_like(validity))
                g_loss.backward()
                self.optimizer_G.step()

                self.optimizer_D.zero_grad()
                real_validity = self.discriminator(data_tensor)
                fake_validity = self.discriminator(generated_data.detach())
                real_loss = self.criterion(real_validity, torch.ones_like(real_validity))
                fake_loss = self.criterion(fake_validity, torch.zeros_like(fake_validity))
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                logging.info(f"Epoch {epoch+1}/100: Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")
            except Exception as e:
                logging.error(f"Error during training at epoch {epoch+1}: {e}")
                raise

    def generate(self):
        try:
            noise = torch.randn(1000, 100)  # Generate 1000 samples
            synthetic_data = self.generator(noise).detach().numpy()
            logging.info("Synthetic data generated.")
        except Exception as e:
            logging.error(f"Error during data generation: {e}")
            raise
        return pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(synthetic_data.shape[1])])

    def save(self, path):
        try:
            torch.save(self.generator.state_dict(), path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Error while saving the model: {e}")
            raise

    def load(self, path):
        try:
            self.generator.load_state_dict(torch.load(path, weights_only=False))
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Error while loading the model: {e}")
            raise
