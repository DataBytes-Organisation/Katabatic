import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
from random import choice
import os
import tempfile

# ---------- MEG Model Classes ----------

class MappingNetwork(nn.Module):
    def __init__(self, kge_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, kge):
        return self.net(kge)

class OriginalMEG(nn.Module):
    def __init__(self, input_dim, kge_dim, embed_dim):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim)
        self.kge_mapper = MappingNetwork(kge_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, input_dim)
        )

    def forward(self, x, kge):
        x_embed = self.token_embed(x)
        kge_embed = self.kge_mapper(kge)
        fused = x_embed + kge_embed
        encoded = self.encoder(fused.unsqueeze(0)).squeeze(0)
        return self.decoder(encoded)

# ---------- Preprocessing and KGE ----------

def preprocess_data(df):
    target_col = df.columns[-1]
    cat_cols = [col for col in df.select_dtypes(include='object').columns if col != target_col]

    if not cat_cols:
        cat_cols = [col for col in df.columns if col != target_col and df[col].nunique() <= 15]

    encoders = {}
    for col in cat_cols + [target_col]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, cat_cols, df, scaler, encoders

def build_kge_matrix(df, cat_cols, embedding_dim=64, walk_length=10, num_walks=20, window=3):
    if not cat_cols:
        return np.zeros((len(df), embedding_dim))

    G = nx.Graph()
    for col in cat_cols:
        for val in df[col].unique():
            G.add_node(f"{col}:{int(val)}")
    for _, row in df[cat_cols].iterrows():
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                G.add_edge(f"{cat_cols[i]}:{int(row[cat_cols[i]])}", 
                           f"{cat_cols[j]}:{int(row[cat_cols[j]])}")

    def random_walk(g, start, wl):
        walk = [start]
        for _ in range(wl - 1):
            nbrs = list(g.neighbors(walk[-1]))
            walk.append(choice(nbrs) if nbrs else walk[-1])
        return walk

    vocab = list(G.nodes())
    vocab_index = {n: i for i, n in enumerate(vocab)}
    co_matrix = np.zeros((len(vocab), len(vocab)))

    for node in vocab:
        for _ in range(num_walks):
            walk = random_walk(G, node, walk_length)
            for i, tgt in enumerate(walk):
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    if i != j:
                        co_matrix[vocab_index[tgt], vocab_index[walk[j]]] += 1

    safe_dim = min(embedding_dim, min(co_matrix.shape))
    embeddings = PCA(n_components=safe_dim).fit_transform(co_matrix)

    row_kges = []
    for _, row in df.iterrows():
        vecs = []
        for col in cat_cols:
            node = f"{col}:{int(row[col])}"
            vecs.append(embeddings[vocab_index.get(node, 0)])
        avg_vec = np.mean(vecs, axis=0)
        if safe_dim < embedding_dim:
            avg_vec = np.pad(avg_vec, (0, embedding_dim - safe_dim))
        row_kges.append(avg_vec)

    return np.array(row_kges)

# ---------- Train MEG ----------

def train_meg_model(X_real_scaled, kge_matrix, input_dim, kge_dim, embed_dim=128, epochs=10):
    X_tensor = torch.tensor(X_real_scaled, dtype=torch.float32)
    KGE_tensor = torch.tensor(kge_matrix, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, KGE_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    meg = OriginalMEG(input_dim=input_dim, kge_dim=kge_dim, embed_dim=embed_dim)
    opt = torch.optim.Adam(meg.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    meg.train()
    for _ in range(epochs):
        for xb, kb in loader:
            out = meg(xb, kb)
            loss = loss_fn(out, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return meg

# ---------- Generate Synthetic Data ----------

def generate_synthetic_data(meg_model, X_real_scaled, kge_matrix, num_samples=None):
    if num_samples is None:
        num_samples = len(X_real_scaled)
        
    X_tensor = torch.randn(num_samples, X_real_scaled.shape[1], dtype=torch.float32)
    KGE_tensor = torch.tensor(kge_matrix[:num_samples], dtype=torch.float32)
    
    meg_model.eval()
    with torch.no_grad():
        synthetic = meg_model(X_tensor, KGE_tensor)
    return synthetic.numpy()

# ---------- Save Synthetic Dataset ----------

def save_synthetic_data(X_synth, original_df, scaler, encoders, target_col):
    try:
        # Inverse transform the scaled data
        X_original_scale = scaler.inverse_transform(X_synth)
        
        # Create DataFrame with original column names (excluding target)
        synth_df = pd.DataFrame(X_original_scale, columns=original_df.drop(columns=[target_col]).columns)
        
        # Add target column if it exists in encoders
        if target_col in encoders:
            # For demo purposes, we'll randomly sample from original target values
            synth_df[target_col] = np.random.choice(original_df[target_col], size=len(synth_df))
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), 'synthetic_data')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(temp_dir, 'synthetic_data.csv')
        synth_df.to_csv(output_path, index=False)
        
        return output_path
        
    except Exception as e:
        print(f"Error during CSV saving: {e}")
        return None