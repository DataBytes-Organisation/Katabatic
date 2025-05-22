import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class GANBLRPP:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.synthetic_data = None

    def preprocess(self, df):
        df = df.dropna()
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(df.select_dtypes(include=[np.number]))
        return scaled

    def train(self, X, components=5):
        self.pca = PCA(n_components=components)
        self.pca.fit(X)

    def generate(self, n_samples=100):
        sampled = np.random.normal(size=(n_samples, self.pca.n_components_))
        gen_data = self.pca.inverse_transform(sampled)
        self.synthetic_data = self.scaler.inverse_transform(gen_data)
        return pd.DataFrame(self.synthetic_data, columns=[f'feature_{i+1}' for i in range(self.synthetic_data.shape[1])])
