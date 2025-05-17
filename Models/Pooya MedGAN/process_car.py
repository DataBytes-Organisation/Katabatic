# process_car.py
# Usage: python process_car.py car.csv <output file> <"binary"|"count">

import sys
import _pickle as pickle
import numpy as np
import csv

if __name__ == '__main__':
    input_file = sys.argv[1]
    out_file = sys.argv[2]
    binary_count = sys.argv[3]

    if binary_count not in ['binary', 'count']:
        print('You must choose either "binary" or "count".')
        sys.exit()

    print('Reading and processing data...')
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

        data = []
        for row in reader:
            data.append(row)

    print('Building types dictionary...')
    types = {}  # key: feature name (e.g., buying_vhigh), value: index
    sequences = []

    for row in data:
        instance_features = []
        for i, val in enumerate(row[:-1]):  # Exclude class
            key = f"{headers[i]}_{val}"
            if key not in types:
                types[key] = len(types)
            instance_features.append(types[key])
        sequences.append(instance_features)

    print('Constructing the matrix...')
    num_instances = len(sequences)
    num_codes = len(types)
    matrix = np.zeros((num_instances, num_codes), dtype=np.float32)

    for i, features in enumerate(sequences):
        for code in features:
            if binary_count == 'binary':
                matrix[i][code] = 1.
            else:
                matrix[i][code] += 1.

    print('Saving output files...')
    pids = list(range(num_instances))  # just use row index as pseudo-pid
    pickle.dump(pids, open(out_file + '.pids', 'wb'), -1)
    pickle.dump(matrix, open(out_file + '.matrix', 'wb'), -1)
    pickle.dump(types, open(out_file + '.types', 'wb'), -1)

    print('Done!')
