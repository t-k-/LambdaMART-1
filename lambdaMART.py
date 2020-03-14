import numpy as np
from sklearn.tree import DecisionTreeRegressor

#from numba import jit

class LambdaMART:
    """Original paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf"""
    def __init__(self, num_trees=100, max_depth=3, learning_rate=1.0):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.trees = []
        self.gamma = np.zeros((num_trees, 2**(max_depth + 1) - 1))
        self.delta_ndcg_first_dict = {}

    def _idcg(self, relevence):
        relevence = np.sort(relevence)[::-1]
        i = np.arange(1, len(relevence)+1)
        return np.dot((2**relevence - 1), 1/np.log2(i + 1))

    #@jit
    def _calculate_lambda(self, relevence, F, qid):
        order = np.argsort(F)[::-1] + 1
        if qid in self.delta_ndcg_first_dict:
            delta_ndcg_first = self.delta_ndcg_first_dict[qid]
        else:
            idcg = self._idcg(relevence)
            if idcg != 0:
                delta_ndcg_first = (2**relevence - 1) / idcg
            else:
                delta_ndcg_first = np.zeros(len(relevence))
            self.delta_ndcg_first_dict[qid] = delta_ndcg_first
        delta_s_matrix = np.reshape(F, (-1, 1)) - np.reshape(F, (1, -1))
        delta_ndcg_matrix = np.zeros((len(relevence), len(relevence)))
        log_order = 1 / np.log2(1 + order)
        for i in range(len(relevence)):
            for j in range(i+1, len(relevence)):
                if relevence[i] != relevence[j]:
                    log_order_swap = np.copy(log_order)
                    log_order_swap[i], log_order_swap[j] = log_order_swap[j], log_order_swap[i]
                    if relevence[i] > relevence[j]:
                        delta_ndcg_matrix[i, j] = np.dot(delta_ndcg_first, log_order_swap - log_order)
                    else:
                        delta_ndcg_matrix[j, i] = np.dot(delta_ndcg_first, log_order_swap - log_order)
        rho_matrix = 1 / (1 + np.exp(delta_s_matrix))
        abs_delta_ndcg_matrix = np.abs(delta_ndcg_matrix)
        omega_matrix = abs_delta_ndcg_matrix * rho_matrix
        lambda_matrix = -omega_matrix
        omega_matrix *= (1 - rho_matrix)
        lambda_matrix -= lambda_matrix.T
        omega_matrix += omega_matrix.T
        return np.sum(lambda_matrix, axis=0), np.sum(omega_matrix, axis=0)

    def fit(self, X, relevence, qid):
        F = np.zeros(np.shape(X)[0])
        eps = 0.000001
        for k in range(self.num_trees):
            lambda_arr = np.array([])
            omega_arr = np.array([])
            for unique_qid in np.unique(qid):
                qid_lambda, qid_omega = self._calculate_lambda(relevence[qid == unique_qid], 
                                                               F[qid == unique_qid], unique_qid)
                lambda_arr = np.append(lambda_arr, qid_lambda)
                omega_arr = np.append(omega_arr, qid_omega)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, lambda_arr)
            self.trees.append(tree)
            leaves = tree.apply(X)
            for leaf in np.unique(leaves):
                leaf_idx = (leaves == leaf)
                self.gamma[k, leaf] = np.sum(lambda_arr[leaf_idx]) / (np.sum(omega_arr[leaf_idx]) + eps)
                F += self.lr * leaf_idx * self.gamma[k, leaf]

    def predict(self, X):
        F = np.zeros(np.shape(X)[0])
        for k in range(len(self.trees)):
            leaves = self.trees[k].apply(X)
            for leaf in np.unique(leaves):
                leaf_idx = (leaves == leaf)
                F += self.lr * leaf_idx * self.gamma[k, leaf]
        return F

if __name__ == '__main__':
    from sklearn.datasets import load_svmlight_file
    train_features, train_rel, train_qid = load_svmlight_file('data/t.txt', query_id = True)
    test_features, test_rel = load_svmlight_file('data/test.txt')
    # Train
    model = LambdaMART(num_trees=100, max_depth=10, learning_rate=0.1)
    model.fit(train_features, train_rel, train_qid)
    # Test
    preds = model.predict(test_features)
    for i, j in zip(preds, test_rel):
        print(i, j)
