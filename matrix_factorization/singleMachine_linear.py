
'''
Naive implmentation of the linear matrix factorization
'''

from pyspark import SparkContext
import sys
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy
from pylab import *
import matplotlib.pyplot as plt

def CSV_to_sparse(netflix_file):
    row_indices = []
    col_indices = []
    data_rating = []

    lines = netflix_file.collect()
    for line in lines:
        line_array = line.split(",")
        row_indices.append(int(line_array[0]) - 1)
        col_indices.append(int(line_array[1]) - 1)
        data_rating.append(float(line_array[2]))
    return csr_matrix((data_rating, (row_indices, col_indices)))

def matrix_factorization(R, P, Q, K, steps=200, alpha=0.0002, beta=0.02):
    iterative_time = list()
    error = list()
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        iterative_time.append(step)
        error.append(e)
        print "iterative steps: ", e, step
        # if e < 0.001:
        #     break
    plt.plot(iterative_time, error)
    savefig('single_200_mf.png', bbox_inches='tight')
    return P, Q.T

if __name__ == "__main__":
    sc = SparkContext(appName="Linear MF")
    netflix_file = sc.textFile("sample_data.csv")
    sparse_data = CSV_to_sparse(netflix_file)
    dense = numpy.asarray(sparse_data.todense())
    training_data = list(dense)
    R = numpy.array(training_data)

    N = len(R)
    M = len(R[0])
    K = 20

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    print "Result of P and Q are:"
    print nP
    print nQ