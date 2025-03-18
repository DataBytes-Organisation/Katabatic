import numpy as np
#import networkx as nx
from pyitlib import discrete_random_variable as drv
import warnings
warnings.filterwarnings("ignore")

def build_graph(X, y, k=2):
  '''
  kDB algorithm

  Param:
  ----------------------
    
  Return:
  ----------------------
  graph edges
  '''
  #ensure data
  num_features = X.shape[1]
  x_nodes = list(range(num_features))
  y_node  = num_features

  #util func
  _x = lambda i:X[:,i]
  _x2comb = lambda i,j:(X[:,i], X[:,j])

  #feature indexes desc sort by mutual information
  sorted_feature_idxs = np.argsort([
    drv.information_mutual(_x(i), y) 
    for i in range(num_features)
  ])[::-1]

  #start building graph
  edges = []
  for iter, target_idx in enumerate(sorted_feature_idxs):
    target_node = x_nodes[target_idx]
    edges.append((y_node, target_node))

    parent_candidate_idxs = sorted_feature_idxs[:iter]
    if iter <= k:
      for idx in parent_candidate_idxs:
        edges.append((x_nodes[idx], target_node))
    else:
      first_k_parent_mi_idxs = np.argsort([
        drv.information_mutual_conditional(*_x2comb(i, target_idx), y)
        for i in parent_candidate_idxs
      ])[::-1][:k]
      first_k_parent_idxs = parent_candidate_idxs[first_k_parent_mi_idxs]

      for parent_idx in first_k_parent_idxs:
        edges.append((x_nodes[parent_idx], target_node))
  return edges

# def draw_graph(edges):
#   '''
#   Draw the graph
# 
#   Param
#   -----------------
#   edges: edges of the graph
# 
#   '''
#   graph = nx.DiGraph(edges)
#   pos=nx.spiral_layout(graph)
#   nx.draw(graph, pos, node_color='r', edge_color='b')
#   nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")


def get_cross_table(*cols, apply_wt=False):
    '''   
    author: alexland

    returns:
      (i) xt, NumPy array storing the xtab results, number of dimensions is equal to 
          the len(args) passed in
      (ii) unique_vals_all_cols, a tuple of 1D NumPy array for each dimension 
          in xt (for a 2D xtab, the tuple comprises the row and column headers)
      pass in:
        (i) 1 or more 1D NumPy arrays of integers
        (ii) if wts is True, then the last array in cols is an array of weights
        
    if return_inverse=True, then np.unique also returns an integer index 
    (from 0, & of same len as array passed in) such that, uniq_vals[idx] gives the original array passed in
    higher dimensional cross tabulations are supported (eg, 2D & 3D)
    cross tabulation on two variables (columns):
    >>> q1 = np.array([7, 8, 8, 8, 5, 6, 4, 6, 6, 8, 4, 6, 6, 6, 6, 8, 8, 5, 8, 6])
    >>> q2 = np.array([6, 4, 6, 4, 8, 8, 4, 8, 7, 4, 4, 8, 8, 7, 5, 4, 8, 4, 4, 4])
    >>> uv, xt = xtab(q1, q2)
    >>> uv
      (array([4, 5, 6, 7, 8]), array([4, 5, 6, 7, 8]))
    >>> xt
      array([[2, 0, 0, 0, 0],
             [1, 0, 0, 0, 1],
             [1, 1, 0, 2, 4],
             [0, 0, 1, 0, 0],
             [5, 0, 1, 0, 1]], dtype=uint64)
      '''
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
      raise ValueError("all arguments must be same size")

    if len(cols) == 0:
      raise TypeError("xtab() requires at least one argument")

    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
      raise ValueError("all input arrays must be 1D")

    if apply_wt:
      cols, wt = cols[:-1], cols[-1]
    else:
      wt = 1

    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = 'float' if apply_wt else 'uint'
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt

def _get_dependencies_without_y(variables, y_name, kdb_edges):
    ''' 
    evidences of each variable without y.

    Param:
    --------------
    variables: variable names

    y_name: class name

    kdb_edges: list of tuple (source, target)
    '''
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t:i for i,(s,t) in enumerate(kdb_edges) if s == y_name}
    for x in variables:
        current_dependencies = [s for s,t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t:mi_desc_order[t] for t in current_dependencies}        
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies

def _add_uniform(array, noise=1e-5):
    ''' 
    if no count on particular condition for any feature, give a uniform prob rather than leave 0
    '''
    sum_by_col = np.sum(array,axis=0)
    zero_idxs = (array == 0).astype(int)
    # zero_count_by_col = np.sum(zero_idxs,axis=0)
    nunique = array.shape[0]
    result = np.zeros_like(array, dtype='float')
    for i in range(array.shape[1]):
        if sum_by_col[i] == 0:
            result[:,i] = array[:,i] + 1./nunique
        elif noise != 0:
            result[:,i] = array[:,i] + noise * zero_idxs[:,i]
        else:
            result[:,i] = array[:,i]
    return result

def _normalize_by_column(array):
    sum_by_col = np.sum(array,axis=0)
    return np.divide(array, sum_by_col,
        out=np.zeros_like(array,dtype='float'),
        where=sum_by_col !=0)

def _smoothing(cct, d):
    '''
    probability smoothing for kdb
    
    Parameters:
    -----------
    cct (np.ndarray): cross count table with shape (x0, *parents)

    d (int): dimension of cct

    Return:
    --------
    smoothed joint prob table
    '''
    #covert cross-count-table to joint-prob-table by doing a normalization alone axis 0
    jpt = _normalize_by_column(cct)
    smoothing_idx = jpt == 0
    if d > 1 and np.sum(smoothing_idx) > 0:
        parent = cct.sum(axis=-1)
        parent = _smoothing(parent, d-1)
        parent_extend = parent.repeat(jpt.shape[-1]).reshape(jpt.shape)
        jpt[smoothing_idx] = parent_extend[smoothing_idx]
    return jpt

def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    '''
    encode the high order feature of X[col] given evidences X[evidence_cols].
    '''
    if evidence_cols is None or len(evidence_cols) == 0:
        return X[:,[col]]
    else:
        evidences = [X[:,_col] for _col in evidence_cols]

        #[1, variable_unique, evidence_unique]
        base = [1, feature_uniques[col]] + [feature_uniques[_col] for _col in evidence_cols[::-1][:-1]]
        cum_base = np.cumprod(base)[::-1]
        
        cols = evidence_cols + [col]
        high_order_feature = np.sum(X[:,cols] * cum_base, axis=1).reshape(-1,1)
        return high_order_feature

def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    '''
    find the constraints infomation for the high order feature X[col] given evidences X[evidence_cols].
    
    Returns:
    ---------------------
    tuple(have_value, high_order_uniques)

    have_value: a k+1 dimensions numpy ndarray of type boolean. 
        Each dimension correspond to a variable, with the order (*evidence_cols, col)
        True indicate the corresponding combination of variable values cound be found in the dataset.
        False indicate not.

    high_order_constraints: a 1d nummy ndarray of type int.
        Each number `c` indicate that there are `c` cols shound be applying the constraints since the last constrant position(or index 0),
        in sequence.         

    '''
    if evidence_cols is None or len(evidence_cols) == 0:
        unique = feature_uniques[col]
        return np.ones(unique,dtype=bool), np.array([unique])
    else:
        cols = evidence_cols + [col]
        cross_table_idxs, cross_table = get_cross_table(*[X[:,i] for i in cols])
        have_value = cross_table != 0
    
        have_value_reshape = have_value.reshape(-1,have_value.shape[-1])
        #have_value_split = np.split(have_value_reshape, have_value_reshape.shape[0], 0)
        high_order_constraints = np.sum(have_value_reshape, axis=-1)
    
        return have_value, high_order_constraints

class KdbHighOrderFeatureEncoder:
    '''
    High order feature encoder that uses the kdb model to retrieve the dependencies between features.
    
    '''
    def __init__(self):
        self.dependencies_ = {}
        self.constraints_ = np.array([])
        self.have_value_idxs_ = []
        self.feature_uniques_ = []
        self.high_order_feature_uniques_ = []
        self.edges_ = []
        self.ohe_ = None
        self.k = None
        #self.full_=True
    
    def fit(self, X, y, k=0):
        '''
        Fit the KdbHighOrderFeatureEncoder to X, y.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            data to fit in the encoder.

        y : array_like of shape (n_samples,)
            label to fit in the encoder.

        k : int, default=0
            k value of the order of the high-order feature. k = 0 will lead to a OneHotEncoder.

        Returns
        -------
        self : object
            Fitted encoder.
        '''
        self.k = k
        edges = build_graph(X, y, k)
        #n_classes = len(np.unique(y))
        num_features = X.shape[1]

        if k > 0:
            dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
        else:
            dependencies = {x:[] for x in range(num_features)}
        
        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X[:,i])) for i in range(num_features)]
        self.edges_ = edges
        #self.full_ = full

        Xk, constraints, have_value_idxs = self.transform(X, return_constraints=True, use_ohe=False)

        from sklearn.preprocessing import OneHotEncoder
        self.ohe_ = OneHotEncoder().fit(Xk)
        self.high_order_feature_uniques_ = [len(c) for c in self.ohe_.categories_]
        self.constraints_ = constraints
        self.have_value_idxs_ = have_value_idxs
        return self
        
    def transform(self, X, return_constraints=False, use_ohe=True):
        """
        Transform X to the high-order features.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to fit in the encoder.
        
        return_constraints : bool, default=False
            Whether to return the constraint informations. 
        
        use_ohe : bool, default=True
            Whether to transform output to one-hot format.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input.
        """
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
        from sklearn.preprocessing import OrdinalEncoder
        Xk = OrdinalEncoder().fit_transform(Xk)
        if use_ohe:
            Xk = self.ohe_.transform(Xk)

        if return_constraints:
            concated_constraints = np.hstack(constraints)
            return Xk, concated_constraints, have_value_idxs
        else:
            return Xk
    
    def fit_transform(self, X, y, k=0, return_constraints=False):
        '''
        Fit KdbHighOrderFeatureEncoder to X, y, then transform X.
        
        Equivalent to fit(X, y, k).transform(X, return_constraints) but more convenient.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            data to fit in the encoder.

        y : array_like of shape (n_samples,)
            label to fit in the encoder.

        k : int, default=0
            k value of the kdb model. k = 0 will lead to a OneHotEncoder.
        
        return_constraints : bool, default=False
            whether to return the constraint informations. 

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input.
        '''
        return self.fit(X, y, k).transform(X, return_constraints)