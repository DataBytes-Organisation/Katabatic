import torch
import torch.nn as nn

class MaskedGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, mask):
        super(MaskedGenerator, self).__init__()
        self.mask = mask
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, x):
        x = self.model(x)
        x = x * torch.tensor(self.mask, dtype=torch.float32)  # Apply the mask to the output
        return x
