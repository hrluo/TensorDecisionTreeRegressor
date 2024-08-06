import numpy as np
L2_norm = lambda A, B: np.linalg.norm(A - B,ord=2)
class Node:
    def __init__(self, predicted_value, leaf_index, split_method=None, samples_X=None, samples_y=None, linear_model=None, cp_model=None, tucker_model=None):
        self.predicted_value = predicted_value
        self.feature_index = None
        self.threshold = None
        self.parent = None
        self.left = None
        self.right = None
        self.split_method = split_method
        self.samples_X = samples_X  # Storing the samples assigned to this node
        self.samples_y = samples_y  # Storing the corresponding labels
        self.linear_model = linear_model
        self.cp_model = cp_model
        self.tucker_model = tucker_model
        self.split_loss = None
        self.leaf_index = leaf_index  # New attribute for leaf nodes
        self.use_mean_as_threshold = True

    def count_child(self):
        if (self.left is None) and (self.right is None):
            return 0
        if (self.left is None) and (self.right is not None):
            return self.right.count_child() + 1
        if (self.left is not None) and (self.right is None):
            return self.left.count_child() + 1
        if (self.left is not None) and (self.right is not None):
            return self.right.count_child() + 1 + self.left.count_child() + 1
        
    def get_depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.get_depth()+1
        
    def get_mean_y(self):
        return np.mean(self.samples_y)

    def get_var_y(self):
        return np.var(self.samples_y)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)
    
    def get_leaves(self):
        res = []
        if self.left is None and self.right is None:
            res.append(self)
            return res
        if self.left.is_leaf():
            res.append(self.left)
        else:
            left_leaves = self.left.get_leaves()
            for L in left_leaves:
                res.append(L)
        if self.right.is_leaf():
            res.append(self.right)
        else:
            right_leaves = self.right.get_leaves()
            for L in right_leaves:
                res.append(L)
        return res

class custom_kmeans:
    def __init__(self, n_clusters, dist_func=L2_norm, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.matshape = None
    
    def fit(self, X):
        #self.matshape = X[0].shape
        # X is a list of matrices
        
        # randomly initialize the centroids
        centroids = [X[i] for i in np.random.choice(len(X), self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # assign each data point to the nearest centroid
            labels = np.array([[self.dist_func(x, c) for c in centroids] for x in X]).argmin(axis=1)
            
            # update the centroids
            new_centroids = [np.mean([x for x, l in zip(X, labels) if l == i], axis=0) for i in range(self.n_clusters)]
            
            # check for convergence
            if np.all([self.dist_func(c, nc) < self.tol for c, nc in zip(centroids, new_centroids)]):
                break
            
            centroids = new_centroids
        
        self.centroids = centroids
    
    def predict(self, X):
        if len(X) == 0:
            return []
        #for x in X:
        #    assert x.shape==self.matshape
        # X is a list of matrices
        
        if self.centroids is None:
            raise ValueError("Must fit the model before predicting.")
        
        # assign each data point to the nearest centroid
        labels = np.array([[self.dist_func(x, c) for c in self.centroids] for x in X]).argmin(axis=1)
        
        return labels

# example usage:
# X = [np.random.rand(2, 2) for _ in range(10)]
# Xnew = [np.random.rand(2, 2) for _ in range(12)]
# n_clusters = 2
#model = custom_kmeans(n_clusters, L2_norm)
#model.fit(X)
#labels = model.predict(X)
#print(labels,model.centroids)

import tensorly as tl
from tensorly.decomposition import parafac,constrained_parafac
from tensorly.decomposition import tucker
from tensorly.regression.cp_regression import CPRegressor
from tensorly.regression.tucker_regression import TuckerRegressor
class TensorDecisionTreeRegressor:
    def __init__(self, max_depth=2, min_samples_split=2, split_method='variance', split_rank=1, CP_reg_rank=None, Tucker_reg_rank=None, kmeans=None, verbose=0, n_mode=None):
        assert n_mode is not None
        self.n_mode = n_mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.split_method = split_method
        self.split_rank = split_rank
        self.CP_reg_rank = self.split_rank
        self.Tucker_reg_rank = self.split_rank
        self.verbose = verbose
        self.use_mean_as_threshold = True#New
        def my_rank(depth,model_type):
            if model_type == 'cp':
                return self.CP_reg_rank
            if model_type == 'tucker':
                if self.n_mode == 3: 
                    return [self.Tucker_reg_rank,self.Tucker_reg_rank]
                elif self.n_mode == 4:
                    return [self.Tucker_reg_rank,self.Tucker_reg_rank,self.Tucker_reg_rank]
                else:
                    raise NotImplementedError('self.Tucker_reg_rank initialization error with self.n_mode=',self.n_mode)
        self.adaptive_rank = my_rank
        #def example_adaptive_rank(depth,model_type):
        #    if model_type == 'cp':
        #        return depth+1
        #    if model_type == 'tucker': 
        #        return [depth+2,depth+1,depth]
        
        if CP_reg_rank is not None:
            self.CP_reg_rank = CP_reg_rank
        else:
            self.CP_reg_rank = self.split_rank
        if Tucker_reg_rank is not None:
            self.Tucker_reg_rank = Tucker_reg_rank        
        else:
            self.CP_reg_rank = self.split_rank
        self.classifier = None
        self.modeltype = 'TensorDecisionTreeRegressor'
        self.support_methods = ['middle','kmeans','variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']#Method selection logic, the middle method is only for testing, it cannot predict.
        self.lowrank_method = 'cp'
        self.modelmaxiter = 200
        self.sample_rate = 0.01#For leverage score sampling
        self.tolerance = 0.1#For branch-and-bound method
        self.leaf_counter = 0  # Initialize leaf counter

    def _rank_k_approx_error(self, X,depth=None):
        if len(X)<=self.min_samples_split:
            return np.inf
        if self.lowrank_method=='cp':
            weights, factors = parafac(X, rank=self.split_rank, l2_reg = np.finfo(np.float32).eps)
            rank_k_approx = tl.cp_to_tensor((weights, factors))
            return tl.norm(X - rank_k_approx)
        if self.lowrank_method=='tucker':
            core, factors = tucker(X, rank=[X.shape[0],self.split_rank,self.split_rank])
            rank_k_approx = tl.tucker_to_tensor((core, factors)) 
            return tl.norm(X - rank_k_approx)
        if self.lowrank_method=='constrained_cp':
            weights, factors = constrained_parafac(X, rank=self.split_rank, l1_reg = 0.1) #To avoid super-sparse tensor
            rank_k_approx = tl.cp_to_tensor((weights, factors))
            return tl.norm(X - rank_k_approx)
            
    def _rank_k_reg_error(self, X, y,depth=None):
        if len(X)<=self.min_samples_split:
            return np.inf
        if self.lowrank_method=='cp':
            if self.adaptive_rank is not None:
                model = CPRegressor(weight_rank=self.adaptive_rank(depth,'cp'),verbose=self.verbose)
            else:
                model = CPRegressor(weight_rank=self.CP_reg_rank,verbose=self.verbose)
            model.fit(X, y)
            y_predict = model.predict(X)
            return tl.norm(y - y_predict)
        if self.lowrank_method=='tucker':
            if self.adaptive_rank is not None:
                model = TuckerRegressor(weight_ranks=self.adaptive_rank(depth,'tucker'), tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=0, verbose=self.verbose)
            else:
                model = TuckerRegressor(weight_ranks=[self.Tucker_reg_rank, self.Tucker_reg_rank], tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=0, verbose=self.verbose)
            model.fit(X, y)
            y_predict = model.predict(X)
            return tl.norm(y - y_predict)
       
    def _split(self, X, y, feature_index, threshold):
        if self.split_method=='middle':
            n = X.shape[0]
            half_n = n // 2
            # create first vector
            less_equal_than_threshold = np.zeros(n)
            less_equal_than_threshold[:half_n] = 1
            less_equal_than_threshold = less_equal_than_threshold.astype(bool)

        if self.split_method in ['variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']:#Method selection logic
            #n = X.shape[0]
            #less_equal_than_threshold = np.zeros(n)
            if X.ndim == 3:
                less_equal_than_threshold = X[:, feature_index[0], feature_index[1]] <= threshold
            elif X.ndim == 4:
                less_equal_than_threshold = X[:, feature_index[0], feature_index[1], feature_index[2]] <= threshold
            else:
                raise NotImplementedError('_split:NotImplementedError:X.ndim=3 or 4')

        if self.split_method=='kmeans':
            #ignore feature_indexs and threshold parameters.
            if X.ndim == 3:
                n_samples, n_features1, n_features2 = X.shape #3-way tensor
                X_list = [X[i, :].reshape(n_features1, n_features2) for i in range(n_samples)]
                self.classifier = custom_kmeans(2,L2_norm)
                self.classifier.fit(X_list)
                labels = self.classifier.predict(X_list)
                less_equal_than_threshold = labels == 0
            elif X.ndim == 4:
                n_samples, n_features1, n_features2, n_features3 = X.shape #4-way tensor
                X_list = [X[i, :].reshape(n_features1, n_features2, n_features3) for i in range(n_samples)]
                self.classifier = custom_kmeans(2,L2_norm)
                self.classifier.fit(X_list)
                labels = self.classifier.predict(X_list)
                less_equal_than_threshold = labels == 0
            else:
                raise NotImplementedError('_split:NotImplementedError:X.ndim=3 or 4')

        greater_than_threshold = ~less_equal_than_threshold
        return less_equal_than_threshold, greater_than_threshold

    def _get_best_split(self, X, y, depth):
        if self.split_method == 'middle':
            less_equal_than_index = np.arange(0,X.shape[0] // 2)
            greater_than_index = np.arange(X.shape[0] // 2,X.shape[0])
            best_feature_index = [less_equal_than_index,greater_than_index]
            #splits = self._split(X, y, feature_index, threshold)
            #var = sum([len(y[split]) * np.var(y[split]) for split in splits])
            #best_feature_index = splits
            best_threshold = None
            best_var = None
            return best_feature_index, best_threshold, best_var
        
        elif self.split_method == 'kmeans':
            return self._split(X, y, [],[]), None, None

        elif self.split_method == 'variance':
            error_type = 'variance'
            optimization_method = 'exhaustive'
            
        elif self.split_method == 'variance_LS':
            error_type = 'variance'
            optimization_method = 'LS'
            
        elif self.split_method == 'variance_BB':
            error_type = 'variance'
            optimization_method = 'BB'
            
        elif self.split_method == 'lowrank':
            error_type = 'low_rank'
            optimization_method = 'exhaustive'
            
        elif self.split_method == 'lowrank_LS':
            error_type = 'low_rank'
            optimization_method = 'LS'
            
        elif self.split_method == 'lowrank_BB':
            error_type = 'low_rank'
            optimization_method = 'BB'
        
        elif self.split_method == 'lowrank_reg':
            error_type = 'low_rank_reg'
            optimization_method = 'exhaustive'
            
        elif self.split_method == 'lowrank_reg_LS':
            error_type = 'low_rank_reg'
            optimization_method = 'LS'
            
        elif self.split_method == 'lowrank_reg_BB':
            error_type = 'low_rank_reg'
            optimization_method = 'BB'
        best_err = np.inf
        best_feature_index = None
        best_threshold = None

        # Define error function based on error_type, compatible with legacy options
        def error_function(subset, subset_y, depth=None):
            if error_type == 'variance':
                return np.var(subset_y)*len(subset_y)
            elif error_type == 'low_rank':
                return self._rank_k_approx_error(subset, depth)
            elif error_type == 'low_rank_reg':
                return self._rank_k_reg_error(subset, subset_y, depth)
            else:
                raise ValueError("Unsupported error type")

        # Define optimization strategy
        if optimization_method == 'exhaustive':
            indices = np.ndindex(X.shape[1:])
             
        elif optimization_method == 'LS':
            shape_of_interest = X.shape[1:]
            total_combinations = np.prod(shape_of_interest)
            sample_size = max(1, int(self.sample_rate * total_combinations))
            variances = np.array([np.var(X[(slice(None),) + indices]) for indices in np.ndindex(shape_of_interest)])
            p_vecs = variances / np.sum(variances)
            #print(p_vecs)
            sampled_indices = np.random.choice(total_combinations, max(1, sample_size), p=p_vecs, replace=False)
            indices = (np.unravel_index(idx, shape_of_interest) for idx in sampled_indices)
            
        elif optimization_method == 'BB':
            bounds = [(0, dim_size - 1) for dim_size in X.shape[1:]]
            queue = [bounds]
            evaluated_combinations = set()
            search_counter = 0
            while queue:
                search_counter = search_counter + 1
                current_bounds = queue.pop(0)
                mid_feature_index = tuple((b[0] + b[1]) // 2 for b in current_bounds)

                if mid_feature_index not in evaluated_combinations:
                    feature_values = X[(slice(None),) + mid_feature_index]
                    evaluated_combinations.add(mid_feature_index)

                    if self.use_mean_as_threshold:
                        # Use the mean value as the threshold
                        threshold = np.mean(feature_values)
                        candidate_idx_left = feature_values <= threshold
                        candidate_idx_right = feature_values > threshold
                        cur_err = error_function(X[candidate_idx_left,...],y[candidate_idx_left], depth) + error_function(X[candidate_idx_right,...],y[candidate_idx_right], depth)

                        if cur_err < best_err:
                            best_err = cur_err
                            best_feature_index = mid_feature_index
                            best_threshold = threshold
                    else:
                        # Exhaustive search over all unique feature values as thresholds
                        for threshold in np.unique(feature_values):
                            candidate_idx_left = feature_values <= threshold
                            candidate_idx_right = feature_values > threshold
                            cur_err = error_function(X[candidate_idx_left,...],y[candidate_idx_left], depth) + error_function(X[candidate_idx_right,...],y[candidate_idx_right], depth)

                            if cur_err < best_err:
                                best_err = cur_err
                                best_feature_index = mid_feature_index
                                best_threshold = threshold

                # Split bounds if larger than tolerance
                for i, bound in enumerate(current_bounds):
                    if bound[1] - bound[0] > self.tolerance:
                        mid_point = (bound[0] + bound[1]) // 2
                        left_bounds = current_bounds.copy()
                        right_bounds = current_bounds.copy()
                        left_bounds[i] = (bound[0], mid_point)
                        right_bounds[i] = (mid_point + 1, bound[1])
                        queue.append(left_bounds)
                        queue.append(right_bounds)
                        break
            print('search size:',search_counter)
        else:
            raise ValueError(f"Unsupported optimization method: {optimization_method} or error function {error_type}.")

        # Loop over selected indices for the chosen optimization strategy
        if optimization_method in ['exhaustive', 'LS']:
            search_counter = 0
            for feature_index in indices:
                search_counter = search_counter + 1
                feature_values = X[(slice(None),) + feature_index]
                if self.use_mean_as_threshold:
                    threshold = np.mean(feature_values)
                    candidate_idx_left = feature_values <= threshold
                    candidate_idx_right = feature_values > threshold
                    cur_err = error_function(X[candidate_idx_left,...],y[candidate_idx_left], depth) + error_function(X[candidate_idx_right,...],y[candidate_idx_right], depth)
                    #print(optimization_method, cur_err,np.sum(candidate_idx_left),'/',np.sum(candidate_idx_right))
                    
                    if cur_err < best_err:
                        best_err = cur_err
                        best_feature_index = feature_index
                        best_threshold = threshold
                else:
                    thresholds = np.unique(feature_values)
                    for threshold in thresholds:
                        candidate_idx_left = feature_values <= threshold
                        candidate_idx_right = feature_values > threshold
                        if len(candidate_idx_left) < self.min_samples_split or len(candidate_idx_right) < self.min_samples_split:
                            continue
                        cur_err = error_function(X[candidate_idx_left,...],y[candidate_idx_left], depth) + error_function(X[candidate_idx_right,...],y[candidate_idx_right], depth)
                        if cur_err < best_err:
                            best_err = cur_err
                            best_feature_index = feature_index
                            best_threshold = threshold
            print('search size:',search_counter)
        print(best_feature_index, best_threshold, best_err)
        return best_feature_index, best_threshold, best_err
            

    def _build_tree(self, X, y, depth=0):
        if len(y) >= self.min_samples_split and depth < self.max_depth:
            feature_index, threshold, loss = self._get_best_split(X, y, depth)
            if feature_index is None:
                if self.n_mode == 3:
                    feature_index = (0,0)
                if self.n_mode == 4:
                    feature_index = (0,0,0)
            if threshold is None:
                if self.n_mode == 3:
                    threshold = X[0,0,0]
                if self.n_mode == 4:
                    threshold = X[0,0,0,0]
                print('None-->X',X.shape,X)
            #print('_build_tree',feature_index,threshold)
            node = Node(predicted_value=np.nan, samples_X=X, samples_y=y, leaf_index=self.leaf_counter)
            if (feature_index is not None) and (threshold is not None):
                #print('_build_tree:branch1')
                splits = self._split(X, y, feature_index, threshold)
                #print('build_tree=',splits)#splits is in the format of less_equal_than_threshold, greater_than_threshold
                self.leaf_counter = self.leaf_counter + 1
                node.root = self.root
                node.split_loss = loss
                node.feature_index = feature_index
                node.threshold = threshold
                node.left = self._build_tree(X[splits[0]], y[splits[0]], depth+1)
                if node.left is not None:
                    node.left.parent = node
                node.right = self._build_tree(X[splits[1]], y[splits[1]], depth+1)
                if node.right is not None:
                    node.right.parent = node
                #return node
            #else:
                #raise NotImplementedError('Unhandled Error: _build_tree (X/y shapes, depth)', X.shape, y.shape, depth,'feature_index', feature_index, 'threshold', threshold)
                # Train both CP and Tucker models at the leaf node
                #print('Xs',X.shape)
            cp_model = None
            tucker_model = None
            node.predicted_value = None
            if (node.left is None) and (node.right is None):
                node.predicted_value = np.mean(y)
                if self.adaptive_rank is not None and X.shape[0]>=self.adaptive_rank(depth,'cp'):
                    cp_model = CPRegressor(weight_rank=self.adaptive_rank(depth,'cp'),verbose=self.verbose)
                else:
                    cp_model = CPRegressor(weight_rank=self.CP_reg_rank,verbose=self.verbose)
                cp_model.fit(X, y)
                if self.adaptive_rank is not None:
                    tucker_model = TuckerRegressor(weight_ranks=self.adaptive_rank(depth,'tucker'), tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
                else:
                    tucker_model = TuckerRegressor(weight_ranks=self.Tucker_reg_rank, tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
                tucker_model.fit(X, y)
            self.leaf_counter = self.leaf_counter + 1
            node.cp_model=cp_model
            node.tucker_model=tucker_model
            node.leaf_index=self.leaf_counter
            return node
        else:
            #return None
            #print('_build_tree:branch2')
            # Train both CP and Tucker models at the leaf node
            if self.adaptive_rank is not None:
                cp_model = CPRegressor(weight_rank=self.adaptive_rank(depth,'cp'),verbose=self.verbose)
            else:
                cp_model = CPRegressor(weight_rank=self.CP_reg_rank,verbose=self.verbose)
            cp_model.fit(X, y)
            if self.adaptive_rank is not None:
                tucker_model = TuckerRegressor(weight_ranks=self.adaptive_rank(depth,'tucker'), tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
            else:
                tucker_model = TuckerRegressor(weight_ranks=self.Tucker_reg_rank, tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
            tucker_model.fit(X, y)
            
            #print(cp_model,tucker_model)
            self.leaf_counter = self.leaf_counter + 1
            return Node(predicted_value=np.mean(y), samples_X=X, samples_y=y, cp_model=cp_model, tucker_model=tucker_model, leaf_index=self.leaf_counter)

    def fit(self, X, y):
        if X.ndim == 3:
            self.n_mode = 3
            self.mode1, self.mode2, self.mode3 = X.shape
        if X.ndim == 4:
            self.n_mode = 4
            self.mode1, self.mode2, self.mode3, self.mode4 = X.shape
        self.root = self._build_tree(X, y)

    def predict(self, X, regression_method='mean'):
        #print('predict->regression_method:',regression_method)
        if regression_method == 'mean':
            return np.array([self._traverse_tree(x, self.root) for x in X])
        elif regression_method == 'cp':
            return np.array([self._traverse_tree_with_cp_regression(x, self.root, 0) for x in X])
        elif regression_method == 'tucker':
            return np.array([self._traverse_tree_with_tucker_regression(x, self.root, 0) for x in X])
        else:
            raise ValueError("Invalid regression method. Expected 'mean', 'cp', or 'tucker'.")
        
    def apply(self, X):
        #This must be performed after the fit method, when self.mode1,2,3 have all been set up
        if X.ndim==2:#Only one observation leaf
            return np.array([self._traverse_tree(X.reshape(self.mode2,self.mode3), self.root, return_leaf_index=True)])
        elif X.ndim==3:
            return np.array([self._traverse_tree(X[i,:,:].reshape(-1,self.mode2,self.mode3).squeeze(), self.root, return_leaf_index=True) for i in range(self.mode1)])
        elif X.ndim==4:
            return np.array([self._traverse_tree(X[i,:,:,:].reshape(-1,self.mode2,self.mode3,self.mode4).squeeze(), self.root, return_leaf_index=True) for i in range(self.mode1)])
        else:
            raise RuntimeError("Invalid and unhandled dimension issue : arise in apply method.")

    def _traverse_tree(self, x, node, return_leaf_index=False):
        if node is None:
            return 0.
        if node.left is None and node.right is None:
            return node.leaf_index if return_leaf_index else np.mean(node.samples_y)
        # Splitting logic remains the same
        if self.split_method in ['middle','variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']:#Method selection logic
            #print('# Splitting logic remains the same_x=',x.shape,x)
            if x.ndim == 3-1:#-1 since we assume x is a single observation.
                if x[node.feature_index[0], node.feature_index[1]] <= node.threshold:
                    return self._traverse_tree(x, node.left, return_leaf_index)
                else:
                    return self._traverse_tree(x, node.right, return_leaf_index)
            elif x.ndim == 4-1:
                if x[node.feature_index[0], node.feature_index[1], node.feature_index[2]] <= node.threshold:
                    return self._traverse_tree(x, node.left, return_leaf_index)
                else:
                    return self._traverse_tree(x, node.right, return_leaf_index)
            else:
                raise NotImplementedError('_traverse_tree: x.ndim=3 or 4, but',x.ndim)
        if self.split_method == 'kmeans':
            labels = self.classifier.predict([x])
            if labels == 0:
                return self._traverse_tree(x, node.left, return_leaf_index)
            else:
                return self._traverse_tree(x, node.right, return_leaf_index)

    def _traverse_tree_with_cp_regression(self, x, node, depth):
        if node is None:
            return np.nan  # Return a suitable fallback value if there is no model
        if (node.left is None and node.right is None) or (node.samples_X.shape[0] < self.min_samples_split):
            if node.cp_model is None:
                if self.adaptive_rank is not None:
                    cp_model = CPRegressor(weight_rank=self.adaptive_rank(depth,'cp'),verbose=self.verbose)
                else:
                    cp_model = CPRegressor(weight_rank=self.CP_reg_rank,verbose=self.verbose)
                cp_model.fit(node.samples_X, node.samples_y) 
                node.cp_model = cp_model
            return node.cp_model.predict(x.reshape(1, *x.shape))[0]
            #print('No cp_model trained, using average value as prediction.')
            #return node.predicted_value  # Fallback to the mean value if the linear model is not available
        if self.split_method in ['middle','variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']:#Method selection logic
            if x.ndim == 3-1:#-1 since we assume x is a single observation.
                if x[node.feature_index[0], node.feature_index[1]] <= node.threshold:
                    return self._traverse_tree_with_cp_regression(x, node.left, depth+1)
                else:
                    return self._traverse_tree_with_cp_regression(x, node.right, depth+1)
            elif x.ndim == 4-1:
                if x[node.feature_index[0], node.feature_index[1], node.feature_index[2]] <= node.threshold:
                    return self._traverse_tree_with_cp_regression(x, node.left, depth+1)
                else:
                    return self._traverse_tree_with_cp_regression(x, node.right, depth+1)
            else:
                raise NotImplementedError('_traverse_tree_with_cp_regression: x.ndim=3 or 4, but',x.ndim)
        if self.split_method=='kmeans':
            #print('kmeans',x.shape)#x is a single obersvation.
            labels = self.classifier.predict([x])
            if labels == 0:
                return self._traverse_tree_with_cp_regression(x, node.left, depth+1)
            else:
                return self._traverse_tree_with_cp_regression(x, node.right, depth+1)

    def _traverse_tree_with_tucker_regression(self, x, node, depth):
        if node is None:
            return np.nan
        if (node.left is None and node.right is None) or (node.samples_X.shape[0] < self.min_samples_split):
            if node.tucker_model is None:
                if self.adaptive_rank is not None:
                    tucker_model = TuckerRegressor(weight_ranks=self.adaptive_rank(depth,'tucker'), tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
                else:
                    tucker_model = TuckerRegressor(weight_ranks=self.Tucker_reg_rank, tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
                tucker_model.fit(node.samples_X, node.samples_y)
                node.tucker_model = tucker_model
            return node.tucker_model.predict(x.reshape(1, *x.shape))[0]
            #print('No tucker_model trained, using average value as prediction.')
            #return node.predicted_value
        if self.split_method in ['middle','variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']:#Method selection logic
            if x.ndim == 3-1:#-1 since we assume x is a single observation.
                if x[node.feature_index[0], node.feature_index[1]] <= node.threshold:
                    return self._traverse_tree_with_tucker_regression(x, node.left, depth+1)
                else:
                    return self._traverse_tree_with_tucker_regression(x, node.right, depth+1)
            elif x.ndim == 4-1:
                if x[node.feature_index[0], node.feature_index[1], node.feature_index[2]] <= node.threshold:
                    return self._traverse_tree_with_tucker_regression(x, node.left, depth+1)
                else:
                    return self._traverse_tree_with_tucker_regression(x, node.right, depth+1)
            else:
                raise NotImplementedError('_traverse_tree_with_tucker_regression: x.ndim=3 or 4, but',x.ndim)
        if self.split_method=='kmeans':
            #print('kmeans',x.shape)#x is a single obersvation.
            labels = self.classifier.predict([x])
            if labels == 0:
                return self._traverse_tree_with_tucker_regression(x, node.left, depth+1)
            else:
                return self._traverse_tree_with_tucker_regression(x, node.right, depth+1)
    def prune(self, X_val=None, y_val=None, model_type='mean',alpha=0.5,depth=0):
        '''Prune the decision tree using cost complexity pruning.'''
        if X_val is None:
            X_val = self.root.samples_X
        if y_val is None:
            y_val = self.root.samples_y  
        self._prune_node(self.root, X_val, y_val, model_type=model_type, alpha=alpha,depth=0)

    def _prune_node(self, node, X_val, y_val, model_type, alpha, depth):
        '''Recursively prune the tree.'''
        if node.is_leaf():
            return
        print('pruning a depth=',depth)
        # Prune left and right children first
        if node.left is not None:
            self._prune_node(node.left, X_val, y_val, model_type, alpha, depth+1)
        if node.right is not None:
            self._prune_node(node.right, X_val, y_val, model_type, alpha, depth+1)

        # Check if this node is a candidate for pruning
        if node.left is not None and node.right is not None:
            complexity_before_pruning = self._compute_complexity_measure(model_type=model_type, depth=depth, alpha=alpha)#(X_val, y_val)

            # Temporarily prune the node
            left, right = node.left, node.right
            node.left, node.right = None, None

            complexity_after_pruning = self._compute_complexity_measure(model_type=model_type, depth=depth, alpha=alpha)#(X_val, y_val)

            # Keep the node pruned if complexity is reduced
            if complexity_after_pruning <= complexity_before_pruning:
                node.left, node.right = None, None
            else:  # Revert pruning
                node.left, node.right = left, right

    #def _compute_complexity_measure(self, X, y, alpha=0.5):
    #    '''Compute a measure of complexity for the tree.'''
    #    predictions = self.predict(X)
    #    mse = L2_norm(y, predictions)
    #    n_leaves = self.count_leaves()
    #    return mse * X.shape[0] + alpha * n_leaves
        
    def _compute_complexity_measure(self, model_type, depth, alpha=0.5):
        '''Compute a measure of complexity for the tree considering model error at each leaf.'''
        
        def _mean_error(self,leaf,depth):
            '''Compute error for leaf nodes using mean value.'''
            predicted_value = np.mean(leaf.samples_y)
            return np.mean((leaf.samples_y - predicted_value) ** 2)

        def _cp_model_error(self,leaf,depth):
            '''Compute error for leaf nodes using CP model.'''
            if self.adaptive_rank is not None:
                cp_model = CPRegressor(weight_rank=self.adaptive_rank(depth,'cp'),verbose=self.verbose)
            else:
                cp_model = CPRegressor(weight_rank=self.CP_reg_rank,verbose=self.verbose)
            cp_model.fit(leaf.samples_X, leaf.samples_y)
            
            predictions = cp_model.predict(leaf.samples_X)
            return np.mean((leaf.samples_y - predictions) ** 2)

        def _tucker_model_error(self,leaf,depth):
            '''Compute error for leaf nodes using Tucker model.'''
            if self.adaptive_rank is not None:
                tucker_model = TuckerRegressor(weight_ranks=self.adaptive_rank(depth,'tucker'), tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
            else:
                tucker_model = TuckerRegressor(weight_ranks=self.Tucker_reg_rank, tol=10e-7, n_iter_max=self.modelmaxiter, reg_W=1, verbose=self.verbose)
            tucker_model.fit(leaf.samples_X, leaf.samples_y)
            
            predictions = tucker_model.predict(leaf.samples_X)
            return np.mean((leaf.samples_y - predictions) ** 2)
            
        leaf_nodes = self.root.get_leaves()
        total_error = 0
    
        for leaf in leaf_nodes:
            if model_type == 'mean':
                error = _mean_error(self,leaf,depth)
            elif model_type == 'cp':
                error = _cp_model_error(self,leaf,depth)
            elif model_type == 'tucker':
                error = _tucker_model_error(self,leaf,depth)
            else:
                raise ValueError(f"Unknown regression method: {leaf.regression_method}")
    
            total_error += error

        n_leaves = self.count_leaves()
        return total_error + alpha * n_leaves

    def count_leaves(self, node=None):
        """Count the number of leaf nodes in the tree."""
        if node is None:
            node = self.root

        if node.is_leaf():
            return 1  # A leaf node contributes 1 to the count
        else:
            # Count leaves in both left and right subtrees and sum them up
            left_count = self.count_leaves(node.left) if node.left is not None else 0
            right_count = self.count_leaves(node.right) if node.right is not None else 0
            return left_count + right_count
        
    def print_tree(self, node=None, indent=" "):
        if node is None:
            node = self.root
        if self.split_method in ['middle','variance','variance_LS','lowrank','lowrank_reg','lowrank_LS','lowrank_reg_LS','lowrank_BB','lowrank_reg_BB']:#Method selection logic
            if node.left is not None:
                if node.samples_X.ndim == 3:
                    print(indent, "if X[:,", node.feature_index[0], ",", node.feature_index[1], "] <= ", node.threshold)
                    self.print_tree(node.left, indent + indent)
                    print(indent, "else: # if X[:,", node.feature_index[0], ",", node.feature_index[1], "] > ", node.threshold)
                    self.print_tree(node.right, indent + indent)
                if node.samples_X.ndim == 4:
                    print(indent, "if X[:,", node.feature_index[0], ",", node.feature_index[1], ",", node.feature_index[2],"] <= ", node.threshold)
                    self.print_tree(node.left, indent + indent)
                    print(indent, "else: # if X[:,", node.feature_index[0], ",", node.feature_index[1], ",", node.feature_index[2], "] > ", node.threshold)
                    self.print_tree(node.right, indent + indent)
            else:
                #print(indent, "return ", node.predicted_value)
                print(indent, "has ", node.count_child()," child nodes, and ",node.samples_X.shape[0]," samples.")
        elif self.split_method == 'kmeans':
            if (node.left is not None) and (node.right is not None):
                print(indent, "if classifier.predict(X) == 0")
                self.print_tree(node.left, indent + indent)
                print(indent, "else: # if classifier.predict(X) == 1")
                self.print_tree(node.right, indent + indent)
            else:
                print(indent, "reaches a leaf, and ",node.samples_X.shape[0]," samples.")

    def _get_leaf_samples_count(self, node):
        if node.left is None and node.right is None:
            return [node.samples_X.shape[0]]
        counts = []
        if node.left is not None:
            counts.extend(self._get_leaf_samples_count(node.left))
        if node.right is not None:
            counts.extend(self._get_leaf_samples_count(node.right))
        return counts

    def get_design_matrix(self):
        leaf_samples_count = self._get_leaf_samples_count(self.root)
        return np.diag(leaf_samples_count)

    def _get_leaf_coefs(self, node):
        if node.left is None and node.right is None:
            if node.cp_model is not None and node.tucker_model is not None:
                cp_coefs = node.cp_model.weight_tensor_
                tucker_coefs = node.tucker_model.weight_tensor_
                return [(cp_coefs, tucker_coefs)]
            else:
                return [None]
        coefs = []
        if node.left is not None:
            coefs.extend(self._get_leaf_coefs(node.left))
        if node.right is not None:
            coefs.extend(self._get_leaf_coefs(node.right))
        return coefs

    def get_fitted_coefs(self):
        return self._get_leaf_coefs(self.root)
        
    def get_depth(self, node=None):
        """Get the maximum depth of the tree."""
        if node is None:
            node = self.root

        if node is None or node.is_leaf():
            return 0
        else:
            # Calculate the depth of the left and right subtrees
            left_depth = self.get_depth(node.left)
            right_depth = self.get_depth(node.right)
            # The depth of the node is 1 + the maximum of the depths of its subtrees
            return 1 + max(left_depth, right_depth)

#Complexity measure C_{\alpha}(T) for regression (work for both types of models)
def Q_m(model):
    res = 0
    model_leaves = model.root.get_leaves()
    
    for L in model_leaves:
        print(L,L.samples_y)
        res = res + L.get_var_y()
    return res
#Formula (9.16) in ESL2
def C_alpha(model,alpha = 1.):
    measure = 0
    Q_m_T = Q_m(model)
    model_leaves = model.root.get_leaves()
    T_size = len(model_leaves)
    for L in model_leaves:
        N_m = L.samples_X.shape[0]
        measure = measure + N_m*Q_m_T + alpha*T_size
    return measure
#Test on 
#C_alpha(model)

#Complexity measure C_tl_{\alpha}(T) for regression using low-rank approximation error (work for both types of models)
def Q_m_tl(model,method='cp'):
    res = 0
    model_leaves = model.root.get_leaves()
    for L in model_leaves:
        #print(L.samples_X.shape)
        m_method = model.lowrank_method
        model.lowrank_method = method
        res = res + model._rank_k_approx_error(L.samples_X,L.get_depth())
        #res = res + model._rank_k_reg_error(L.samples_X,L.samples_y,L.get_depth())
        model.lowrank_method = m_method
    return res
#Mimicing formula (9.16) in ESL2
def C_alpha_tl(model,method,alpha = 1.):
    measure = 0
    Q_m_T = Q_m_tl(model,method=method)
    #print(Q_m_T)
    model_leaves = model.root.get_leaves()
    T_size = len(model_leaves)
    for L in model_leaves:
        N_m = L.samples_X.shape[0]
        measure = measure + N_m*Q_m_T + alpha*T_size
    return measure
#Test on
#C_alpha_tl(model,'cp',0.1)
