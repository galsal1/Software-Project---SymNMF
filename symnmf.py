import sys
import numpy as np
import math
import symnmf_capi

np.random.seed(0)

def handler():
    """"
    a general handler function for errors
    """
    print("An Error Has Occurred")
    exit()


def build_H(k, n, m):
    """
    builds the H matrix ( n * m matrix )
    :param k: - number of dimensions
    :param n: - number of dimensions
    :param m: - mean of the W matrix
    :return uniformally generated H matrix
    """
    H = []
    C = 2 * (np.sqrt(m / k))
    for i in range(n):
        H.append([])
        for j in range(k):
            H[i].append(C*np.random.uniform())
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


def printarr(res):
    """
    outputing the matrix to the console
    :param res: a number represantion of the matrix
    """
    if (type(res[0]) == float):
        print(np.diag(res))
    else:
        formatted_rows = []
        for row in res:
            formatted_row = ''.join(f'{num:.4f},' for num in row)
            formatted_row = formatted_row[0:-1]
            formatted_rows.append(formatted_row)
        print('\n'.join(formatted_rows))


if __name__ == "__main__":
    args = sys.argv[1:]
    options = ["symnmf", "sym", "ddg", "norm"]
    try:
        K = int(args[0])
        goal = str(args[1])
        file = str(args[2])
        X = build_X(file)
        d = len(X[0])
        n = len(X)
    except:
        handler()
    # goal "branching"
    if goal == options[0]:
        W = symnmf_capi.norm(X)
        H = build_H(K, n, np.mean(np.array(W)))
        res = symnmf_capi.symnmf(K, W, H)
    elif goal == options[1]:
        res = symnmf_capi.sym(X)

    elif goal == options[2]:
        res = symnmf_capi.ddg(X)
    elif goal == options[3]:
        res = symnmf_capi.norm(X)
    else:
        handler()
    printarr(res)