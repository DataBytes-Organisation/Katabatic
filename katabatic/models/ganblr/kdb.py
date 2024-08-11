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
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = np.ones(len(cols[0]), dtype=int)

    uniq_vals_all_cols, idx = zip(*(np.unique(col, return_inverse=True) for col in cols))
    shape_xt = [len(uniq_vals) for uniq_vals in uniq_vals_all_cols]
    dtype_xt = 'float' if apply_wt else 'uint'
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt

def get_dependencies_without_y(variables, y_name, kdb_edges):
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}
    
    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            dependencies[x] = sorted(current_dependencies, key=lambda t: mi_desc_order[t])
        else:
            dependencies[x] = current_dependencies
    return dependencies

def add_uniform(array, noise=1e-5):
    sum_by_col = np.sum(array, axis=0)
    zero_idxs = array == 0
    nunique = array.shape[0]
    result = np.where(sum_by_col == 0, 1./nunique, array)
    if noise != 0:
        result += noise * zero_idxs
    return result

def normalize_by_column(array):
    sum_by_col = np.sum(array, axis=0)
    return np.divide(array, sum_by_col, out=np.zeros_like(array, dtype='float'), where=sum_by_col != 0)

def smoothing(cct, d):
    jpt = normalize_by_column(cct)
    smoothing_idx = jpt == 0
    if d > 1 and np.any(smoothing_idx):
        parent = cct.sum(axis=-1)
        parent = smoothing(parent, d-1)
        parent_extend = np.repeat(parent, jpt.shape[-1]).reshape(jpt.shape)
        jpt[smoothing_idx] = parent_extend[smoothing_idx]
    return jpt

def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    if not evidence_cols:
        return X[:, [col]]
    
    evidences = [X[:, _col] for _col in evidence_cols]
    base = [1, feature_uniques[col]] + [feature_uniques[_col] for _col in reversed(evidence_cols[:-1])]
    cum_base = np.cumprod(base)[::-1]
    
    cols = evidence_cols + [col]
    high_order_feature = np.sum(X[:, cols] * cum_base, axis=1, keepdims=True)
    return high_order_feature

def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    if not evidence_cols:
        unique = feature_uniques[col]
        return np.ones(unique, dtype=bool), np.array([unique])
    
    cols = evidence_cols + [col]
    cross_table_idxs, cross_table = get_cross_table(*[X[:, i] for i in cols])
    have_value = cross_table != 0
    
    have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
    high_order_constraints = np.sum(have_value_reshape, axis=-1)
    
    return have_value, high_order_constraints

class KdbHighOrderFeatureEncoder:
    def __init__(self):
        self.dependencies_ = {}
        self.constraints_ = np.array([])
        self.have_value_idxs_ = []
        self.feature_uniques_ = []
        self.high_order_feature_uniques_ = []
        self.edges_ = []
        self.ohe_ = None
        self.k = None
    
    def fit(self, X, y, k=0):
        self.k = k
        edges = build_graph(X, y, k)
        num_features = X.shape[1]

        if k > 0:
            dependencies = get_dependencies_without_y(range(num_features), num_features, edges)
        else:
            dependencies = {x: [] for x in range(num_features)}
        
        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X[:, i])) for i in range(num_features)]
        self.edges_ = edges

        Xk, constraints, have_value_idxs = self.transform(X, return_constraints=True, use_ohe=False)

        self.ohe_ = OneHotEncoder(sparse=False).fit(Xk)
        self.high_order_feature_uniques_ = [len(c) for c in self.ohe_.categories_]
        self.constraints_ = constraints
        self.have_value_idxs_ = have_value_idxs
        return self
        
    def transform(self, X, return_constraints=False, use_ohe=True):
        Xk = []
        have_value_idxs = []
        constraints = []
        for k, v in self.dependencies_.items():
            xk = get_high_order_feature(X, k, v, self.feature_uniques_)
            Xk.append(xk)

            if return_constraints:
                idx, constraint = get_high_order_constraints(X, k, v, self.feature_uniques_)
                have_value_idxs.append(idx)
                constraints.append(constraint)
        
        Xk = np.hstack(Xk)
        if use_ohe:
            Xk = self.ohe_.transform(Xk)

        if return_constraints:
            concated_constraints = np.hstack(constraints)
            return Xk, concated_constraints, have_value_idxs
        else:
            return Xk
    
    def fit_transform(self, X, y, k=0, return_constraints=False):
        self.fit(X, y, k)
        return self.transform(X, return_constraints)
