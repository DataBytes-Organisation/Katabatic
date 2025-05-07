import torch
import torch.optim as optim
from masked_generator import MaskedGenerator
from discriminator import Discriminator

class MEG_Adapter:
    def __init__(self, input_dim, output_dim, num_generators=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_generators = num_generators

        # Initialize generators and discriminator
        self.generators = [MaskedGenerator(input_dim, output_dim, mask=[1]*output_dim) for _ in range(num_generators)]
        self.discriminator = Discriminator(output_dim)

        # Define optimizers and loss function
        self.optimizers = [optim.Adam(generator.parameters(), lr=0.0002) for generator in self.generators]
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss

    def train(self, data, num_epochs=5000, batch_size=64):
        for epoch in range(num_epochs):
            # Create batches from data
            for batch_start in range(0, len(data), batch_size):
                batch_data = data[batch_start:batch_start + batch_size]
                # Prepare real labels and fake labels
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_output = self.discriminator(batch_data)
                loss_real = self.criterion(real_output, real_labels)
                fake_data = [gen(torch.randn(batch_size, self.input_dim)) for gen in self.generators]
                fake_output = self.discriminator(torch.cat(fake_data, dim=1))
                loss_fake = self.criterion(fake_output, fake_labels)
                loss_D = loss_real + loss_fake
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generators
                for generator, optimizer in zip(self.generators, self.optimizers):
                    optimizer.zero_grad()
                    fake_data = generator(torch.randn(batch_size, self.input_dim))
                    fake_output = self.discriminator(fake_data)
                    loss_G = self.criterion(fake_output, real_labels)  # Generator wants to fool the discriminator
                    loss_G.backward()
                    optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

    def generate(self, num_samples):
        noise = torch.randn(num_samples, self.input_dim)
        fake_data = [gen(noise) for gen in self.generators]
        return torch.cat(fake_data, dim=1)
