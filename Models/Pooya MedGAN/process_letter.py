import pandas as pd
import numpy as np
import pickle
import sys
import os

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_file = sys.argv[2]
    data_type = sys.argv[3]

    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_csv)

    # Drop target letter column
    df = df.drop(columns=['letter'])

    print("[INFO] All columns are numeric. No encoding needed.")

    # Convert to float32 for TensorFlow compatibility
    matrix = df.astype(np.float32).values
    print("Shape of matrix:", matrix.shape)

    # Create dummy feature index
    feature_index = {"col_" + str(i): i for i in range(matrix.shape[1])}
    pids = [str(i) for i in range(len(matrix))]

    print("[INFO] Saving files...")
    pickle.dump(matrix, open(output_file + ".matrix", "wb"), -1)
    pickle.dump(feature_index, open(output_file + ".types", "wb"), -1)
    pickle.dump(pids, open(output_file + ".pids", "wb"), -1)
    print("Saved matrix to", output_file)