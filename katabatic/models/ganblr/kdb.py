import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import OneHotEncoder

def build_graph(X, y, k=2):
    num_features = X.shape[1]
    x_nodes = list(range(num_features))
    y_node = num_features

    mutual_info = [drv.information_mutual(X[:, i], y) for i in range(num_features)]
    sorted_feature_idxs = np.argsort(mutual_info)[::-1]

    edges = []
    for iter, target_idx in enumerate(sorted_feature_idxs):
        target_node = x_nodes[target_idx]
        edges.append((y_node, target_node))

        parent_candidate_idxs = sorted_feature_idxs[:iter]
        if iter <= k:
            edges.extend((x_nodes[idx], target_node) for idx in parent_candidate_idxs)
        else:
            conditional_mi = [
                drv.information_mutual_conditional(X[:, i], X[:, target_idx], y)
                for i in parent_candidate_idxs
            ]
            first_k_parent_idxs = parent_candidate_idxs[np.argsort(conditional_mi)[::-1][:k]]
            edges.extend((x_nodes[parent_idx], target_node) for parent_idx in first_k_parent_idxs)
    
    return edges

def get_cross_table(*cols, apply_wt=False):
    if not cols:
        raise ValueError("At least one column is required")
    if not all(len(col) == len(cols[0]) for col in cols):
        raise ValueError("All columns must have the same length")
    
    if apply_wt:
