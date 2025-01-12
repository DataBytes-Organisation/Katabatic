import torch

# Function to save the trained model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Function to load a saved model
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
