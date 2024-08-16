import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .ganblr import GANBLR
from sklearn.ensemble import RandomForestClassifier

class GanblrAlgorithmModel:
    def __init__(self, k=2, batch_size=64, epochs=10):
        self.k = k
        self.batch_size = batch_size
        self.epochs = epochs
        self.ganblr_model = GANBLR()
        self.synthetic_data = None
    
    def preprocess_data(self, X, y):
        """
        Preprocess the input data, ensuring it is in a suitable format for GANBLR.
        """
        # This would involve encoding the data, if not already encoded.
        X = self.ganblr_model._ordinal_encoder.fit_transform(X)
        y = self.ganblr_model._label_encoder.fit_transform(y)
        return X, y
    
    def train_ganblr(self, X, y):
        """
        Train the GANBLR model on the given data.
        """
        X, y = self.preprocess_data(X, y)
        print("Training GANBLR model...")
        self.ganblr_model.fit(X, y, k=self.k, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        print("GANBLR model training completed.")
    
    def generate_synthetic_data(self, size=None):
        """
        Generate synthetic data using the trained GANBLR model.
        """
        print("Generating synthetic data...")
        self.synthetic_data = self.ganblr_model.sample(size=size, verbose=0)
        print(f"Generated {size if size else len(self.synthetic_data)} synthetic samples.")
        return self.synthetic_data
    
    def evaluate_synthetic_data(self, X_test, y_test, model='rf'):
        """
        Evaluate the quality of synthetic data using a classifier model.
        """
        print("Evaluating synthetic data using TSTR...")
        accuracy = self.ganblr_model.evaluate(X_test, y_test, model=model)
        print(f"Evaluation accuracy: {accuracy}")
        return accuracy
    
    def run_full_pipeline(self, X, y, test_size=0.2, evaluation_model='rf'):
        """
        Run the full pipeline: Train GANBLR, generate synthetic data, and evaluate.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train GANBLR
        self.train_ganblr(X_train, y_train)
        
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data(size=len(X_train))
        
        # Evaluate the synthetic data
        accuracy = self.evaluate_synthetic_data(X_test, y_test, model=evaluation_model)
        return accuracy

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('your_dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Initialize the GANBLR algorithm model
    model = GanblrAlgorithmModel(k=2, batch_size=64, epochs=10)
    
    # Run the full pipeline and evaluate
    accuracy = model.run_full_pipeline(X, y, test_size=0.2, evaluation_model='rf')
    print(f"Final Evaluation Accuracy: {accuracy}")
