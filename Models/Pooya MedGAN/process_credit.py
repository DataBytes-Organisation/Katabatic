import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python process_credit.py <credit.csv> <output_prefix> <binary|count>")
        sys.exit()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = sys.argv[3]

    if mode not in ['binary', 'count']:
        print('Choose either "binary" or "count" as the third argument.')
        sys.exit()

    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_file)

    # Drop target column if it exists
    if 'default' in df.columns:
        df.drop('default', axis=1, inplace=True)

    print("[INFO] Handling missing values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(df[col].median())  # Use median instead of mode for numeric

    print("[INFO] Encoding categorical features...")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print("[INFO] Applying OneHotEncoder...")
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df.values)

    matrix = encoded.astype('float32')
    if mode == 'binary':
        matrix = (matrix > 0).astype('float32')
        print("[INFO] Converted matrix to binary format.")

    print("[INFO] Final matrix shape:", matrix.shape)

    feature_index = dict(("col_%d" % i, i) for i in range(matrix.shape[1]))
    pids = list(range(matrix.shape[0]))

    pickle.dump(matrix, open(output_file + '.matrix', 'wb'), -1)
    pickle.dump(feature_index, open(output_file + '.types', 'wb'), -1)
    pickle.dump(pids, open(output_file + '.pids', 'wb'), -1)

    print("[INFO] Done! Files saved as: %s.matrix, %s.types, %s.pids" % (output_file, output_file, output_file))

