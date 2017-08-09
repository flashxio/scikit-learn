import flashpy as fp
import numpy as np
import fplearn
import sklearn

from fplearn.linear_model import Ridge as fp_Ridge
from fplearn.linear_model import ridge_regression as fp_ridge_regression
from fplearn.linear_model import LogisticRegression as fp_LogisticRegression
from fplearn.utils.extmath import row_norms as fp_row_norms
from fplearn.metrics.pairwise import euclidean_distances as fp_euclidean_distances
from fplearn.cluster import k_means as fp_kmeans
from fplearn.model_selection import cross_val_score as fp_cross_val_score
from fplearn.utils.extmath import randomized_svd as fp_randomized_svd

from sklearn.linear_model import Ridge as sk_Ridge
from sklearn.linear_model import ridge_regression as sk_ridge_regression
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.utils.extmath import row_norms as sk_row_norms
from sklearn.metrics.pairwise import euclidean_distances as sk_euclidean_distances
from sklearn.cluster import k_means as sk_kmeans
from sklearn.model_selection import cross_val_score as sk_cross_val_score
from sklearn.utils.extmath import randomized_svd as sk_randomized_svd

fp.init_flashpy()

def verify(fp_arr, np_arr):
    assert fp_arr.ndim == np_arr.ndim
    assert fp_arr.shape == np_arr.shape
    assert fp_arr.size == np_arr.size
    assert fp_arr.itemsize == np_arr.itemsize
    assert fp_arr.nbytes == np_arr.nbytes
    tmp = np.array(fp_arr, copy=True)
    assert np.all(np.absolute(tmp - np_arr) < 1e-10)

print("test ridge_regression")
X = fp.array(np.random.normal(scale=100, size=[1000000, 10]))
y = fp.array(np.random.normal(scale=100, size=1000000))
fp_res = fp_ridge_regression(X, y, 1, solver='svd')
sk_res = sk_ridge_regression(X, y, 1, solver='svd')
verify(fp_res, sk_res)

print("test Ridge")
fp_model = fp_Ridge(solver='svd')
fp_model.fit(X, y)
sk_model = sk_Ridge(solver='svd')
sk_model.fit(X, y)
verify(fp_model.coef_, sk_model.coef_)

print("test cross validation")
res1 = fp_cross_val_score(fp_model, X, y)
res2 = sk_cross_val_score(sk_model, X, y)
verify(res1, res2)

print("test logistic with lbfgs")
fp_model = fp_LogisticRegression(solver='lbfgs')
fp_model.fit(X, y > 50)
sk_model = sk_LogisticRegression(solver='lbfgs')
sk_model.fit(np.array(X), np.array(y > 50))
verify(abs(fp_model.coef_), abs(sk_model.coef_))

#print("test logistic with newton-cg")
#fp_model = fp_LogisticRegression(solver='newton-cg')
#fp_model.fit(X, y > 50)
#sk_model = sk_LogisticRegression(solver='newton-cg')
#sk_model.fit(np.array(X), np.array(y > 50))
#verify(abs(fp_model.coef_), abs(sk_model.coef_))

print("test randomized SVD")
fp_U, fp_s, fp_V = fp_randomized_svd(X.T, 5, power_iteration_normalizer='QR')
sk_U, sk_s, sk_V = sk_randomized_svd(np.array(X).T, 5, power_iteration_normalizer='QR')
verify(abs(fp_U), abs(sk_U))
verify(abs(fp_V), abs(sk_V))
print(fp_s - sk_s)

print("test row_norms")
fp_res = fp_row_norms(X)
sk_res = sk_row_norms(X)
verify(fp_res, sk_res)

print("test euclidean distances")
Y = fp.array(np.random.normal(scale=100, size=[9, 10]))
fp_res = fp_euclidean_distances(X, Y)
sk_res = sk_euclidean_distances(X, Y)
verify(fp_res, sk_res)

idx = np.random.uniform(low=0, high=X.shape[0], size=10).astype('i')
init_centers = X[idx,:]
fp_res = fp_kmeans(X, 10, init=init_centers, n_init=1, max_iter=1, algorithm='full')
sk_res = sk_kmeans(X, 10, init=init_centers, n_init=1, max_iter=1, algorithm='full')
print(fp_res[0] - sk_res[0])
