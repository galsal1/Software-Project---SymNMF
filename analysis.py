import math

import symnmf_capi
import kmeans
import sys
import sklearn.metrics as mt
import numpy as np
np.random.seed(0)

MAX_ITER = 300
def handler():
    """"
    a general handler function for errors
    """
    print("An Error Has Occurred")
    exit()

def matrix_avg(mat):
    """

    :param mat: a matrix to calculate its average
    :return: returns a entry-wise average of a 2D matrix
    """
    entries_num = len(mat[0]) * len(mat)
    count = 0
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            count += mat[row][col]
    avg = count / entries_num if entries_num != 0 else 0
    return avg
def build_H(k, n, m):
    """
    builds the H matrix ( n * m matrix )
    :param k: - number of dimensions
    :param n: - number of dimensions
    :param m: - mean of the W matrix
    :return uniformally generated H matrix
    """
    H = []
    C = 2 * (math.sqrt(m / k))
    for i in range(n):
        row = []
        for j in range(k):
            row.append(np.random.uniform(0, C))
        H.append(row)
    return H
def build_X(file):
    """

    :param file: .txt filepath to a file containing N R^m euclidian dots
    :return: the X matrix
    """
    x = []
    with open(file) as fp:
        for line in fp:
            line = [float(x) for x in line.split(",")]
            x.append(line)
    return x

def Symnmf_request(X, K):
    """
    a wrapper function from symnmf requests not from __main__
    :param X: initial data
    :param K: amount of clusters
    :return: returns H matrix of correlation betwen each center and each vector
    """
    W = symnmf_capi.norm(X)
    H = build_H(K, n, matrix_avg(W))
    res = symnmf_capi.symnmf(K,W,H)
    return res

def compare(X,sym_clusters , kmeans_clusters):
    """
    :param X: the n * d original data
    :param sym_clusters: the clusters achieved by symnmf algorithm
    :param kmeans_clusters: the clusters achieved by K-means algorith
    :return: tuple of silhouette factor of symnmf and k-means respectively
    """
    sym_res = mt.silhouette_score(X,sym_clusters)
    kmeans_res = mt.silhouette_score(X,kmeans_clusters)
    return sym_res, kmeans_res


def get_classification(X,centers):
    """
    gets the group that each vector in X belongs to according to Kmeans
    :param X: the data given by the user
    :param centers: the centers yielded by the Kmeans algorithms
    :return: vector labeling each vector to its center.
    """
    distances = np.zeros((len(X),len(centers)))
    for vector in range(len(X)):
        for center in range(len(centers)):
            dis = np.linalg.norm(np.asmatrix(X[vector]) - np.asmatrix(centers[center]))
            distances[vector][center] = dis
    return distances.argmin(axis=1)


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        file = args[1]
        K = int(args[0])
        X = build_X(file)
        d = len(X[0])
        n = len(X)
    except:
        handler()
    symnmf_clusters = Symnmf_request(X,K)
    kmeans_clusters = kmeans.Kmeans(K,n,d,MAX_ITER,file)
    kmeans_clusters = get_classification(X,kmeans_clusters)
    symnmf_clusters = np.argmax(symnmf_clusters,axis=1)
    symnmf_res , kmeans_res = compare(X,symnmf_clusters,kmeans_clusters)
    print(f"nmf: {symnmf_res:.4f}\nkmeans: {kmeans_res:.4f}")
