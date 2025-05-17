import sys
import _pickle as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python process_chess.py <chess.csv> <output_file_prefix> <binary|count>")
        sys.exit()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = sys.argv[3]

    if mode not in ['binary', 'count']:
        print('Choose "binary" or "count" for mode.')
        sys.exit()

    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_file)

    # Drop label/target if present
    if 'class' in df.columns:
        df.drop('class', axis=1, inplace=True)

    # Treat all columns as categorical
    categorical_cols = df.columns.tolist()
    print("[INFO] Treating all columns as categorical:", categorical_cols)

    print("[INFO] Encoding categorical features...")
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print("[INFO] Applying OneHotEncoder...")
    encoder = OneHotEncoder(sparse=False)
    encoded_cat = encoder.fit_transform(df[categorical_cols])

    matrix = encoded_cat.astype('float32')

    if mode == 'binary':
        matrix = (matrix > 0).astype('float32')
        print("[INFO] Binarized the matrix.")

    print("[INFO] Final matrix shape:", matrix.shape)

    # Build feature index
    feature_index = {"col_{}".format(i): i for i in range(matrix.shape[1])}
    pids = list(range(matrix.shape[0]))

    # Save files
    pickle.dump(matrix, open(output_file + '.matrix', 'wb'), -1)
    pickle.dump(feature_index, open(output_file + '.types', 'wb'), -1)
    pickle.dump(pids, open(output_file + '.pids', 'wb'), -1)

    print("[INFO] Saved:", output_file + ".matrix/.types/.pids")
