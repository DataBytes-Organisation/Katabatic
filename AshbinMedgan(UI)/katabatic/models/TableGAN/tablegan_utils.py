import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

def record_to_matrix(record, matrix_size=32):
    matrix = np.zeros((matrix_size, matrix_size))
    flat_record = record.flatten()
    matrix.flat[:len(flat_record)] = flat_record
    return matrix

def preprocess_data(X, y):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    X_matrices = np.array([record_to_matrix(record) for record in X_scaled])
    X_reshaped = X_matrices.reshape(-1, 32, 32, 1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X_reshaped, y_categorical, scaler, label_encoder

def postprocess_data(generated_data, generated_labels, scaler, label_encoder):
    n_samples = generated_data.shape[0]
    synthetic_flat = generated_data.reshape(n_samples, -1)[:, :scaler.n_features_in_]
    synthetic_original = scaler.inverse_transform(synthetic_flat)
    
    synthetic_labels = label_encoder.inverse_transform(generated_labels.argmax(axis=1))
    
    return np.column_stack((synthetic_original, synthetic_labels))