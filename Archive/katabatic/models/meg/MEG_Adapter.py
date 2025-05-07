import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from MEG_SPI import MEGModelSPI

class MEG_Adapter(MEGModelSPI):
    
    def __init__(self, model_type, num_models, masks, input_dim, output_dim):
        super().__init__(model_type, num_models, masks)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models = []
        self.optimizers = []
        self.criterion = nn.BCELoss()
        self.discriminator = Discriminator(output_dim)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
    
    def load_model(self):

        for mask in self.masking_strategy:
            model = MaskedGenerator(self.input_dim, self.output_dim, mask)
            optimizer = optim.Adam(model.parameters(), lr=0.0002)
            self.models.append(model)
            self.optimizers.append(optimizer)
    
    def load_data(self, data):

        X = data.drop('Purchase', axis=1)
        y = data['Purchase']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    def train(self, num_epochs=5000, batch_size=64):

        for epoch in range(num_epochs):
            self.discriminator.train()
            self.optimizer_D.zero_grad()

            idx = torch.randperm(self.X_train_tensor.size(0))[:batch_size]
            real_data = self.X_train_tensor[idx]
            real_labels = torch.ones((real_data.size(0), 1))


            noise = torch.randn(real_data.size(0), self.input_dim)
            fake_data = torch.zeros_like(real_data)
            for model in self.models:
                fake_data += model(noise)
            fake_labels = torch.zeros((real_data.size(0), 1))


            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data.detach())
            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()


            for i, model in enumerate(self.models):
                self.optimizers[i].zero_grad()
                noise = torch.randn(batch_size, self.input_dim)
                fake_data = model(noise)
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, torch.ones((batch_size, 1)))
                g_loss.backward()
                self.optimizers[i].step()

            if epoch % 500 == 0:
                print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    def generate(self, num_samples):

        synthetic_parts = [model(torch.randn(num_samples, self.input_dim)).detach().numpy() for model in self.models]
        combined_data = np.sum(synthetic_parts, axis=0)
        synthetic_df = pd.DataFrame(combined_data, columns=self.X_train_tensor.columns) 
        return synthetic_df

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class MaskedGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, mask):
        super(MaskedGenerator, self).__init__()
        self.mask = mask
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x) * self.mask
        return output
