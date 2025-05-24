import sys
import _pickle as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python process_adult.py <adult.csv> <output_file_prefix> <binary|count>")
        sys.exit()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = sys.argv[3]

    if mode not in ['binary', 'count']:
        print('Choose "binary" or "count" for mode.')
        sys.exit()

    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_file)

    # Drop columns that shouldn't be encoded
    if 'fnlwgt' in df.columns:
        df.drop('fnlwgt', axis=1, inplace=True)
    if 'class' in df.columns:
        df.drop('class', axis=1, inplace=True)

    # Detect columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()

    print("[INFO] Categorical columns: {}".format(categorical_cols))
    print("[INFO] Numerical columns: {}".format(numerical_cols))

    # Label encode categorical columns to numeric codes
    print("[INFO] Label encoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # One-hot encode the label-encoded categorical columns
    print("[INFO] Applying OneHotEncoder...")
    encoder = OneHotEncoder(sparse=False)
    encoded_cat = encoder.fit_transform(df[categorical_cols])

    # Combine with numeric columns
    print("[INFO] Combining features...")
    matrix = np.hstack((df[numerical_cols].values, encoded_cat)).astype('float32')

    if mode == 'binary':
        matrix = (matrix > 0).astype('float32')
        print("[INFO] Binarized the matrix.")

    print("[INFO] Final matrix shape: {}".format(matrix.shape))

    # Create feature index
    feature_index = {}
    col_idx = 0
    for col in numerical_cols:
        feature_index[col] = col_idx
        col_idx += 1

    onehot_feature_names = ["cat_{}".format(i) for i in range(encoded_cat.shape[1])]
    for col in onehot_feature_names:
        feature_index[col] = col_idx
        col_idx += 1

    # Generate dummy patient IDs
    pids = list(range(matrix.shape[0]))

    # Save files
    pickle.dump(matrix, open(output_file + '.matrix', 'wb'), -1)
    pickle.dump(feature_index, open(output_file + '.types', 'wb'), -1)
    pickle.dump(pids, open(output_file + '.pids', 'wb'), -1)

    print("[INFO] Saved: {}.matrix, .types, .pids".format(output_file))
