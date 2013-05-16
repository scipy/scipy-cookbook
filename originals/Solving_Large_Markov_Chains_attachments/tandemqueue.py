#!/usr/bin/env python

labda, mu1, mu2 = 1., 1.01, 1.001
N1, N2 = 50, 50
size = N1*N2

from numpy import ones, zeros, empty
import scipy.sparse as sp
import  pysparse 
from pylab import matshow, savefig
from scipy.linalg import norm
import time

def state(i,j):
    return j*N1 + i

def fillOffDiagonal(Q):
    # labda
    for i in range(0,N1-1):
        for j in range(0,N2):
            Q[(state(i,j),state(i+1,j))]= labda
    # mu2
    for i in range(0,N1):
        for j in range(1,N2):
            Q[(state(i,j),state(i,j-1))]= mu2
    # mu1
    for i in range(1,N1):
        for j in range(0,N2-1):
            Q[(state(i,j),state(i-1,j+1))]= mu1
    print "ready filling"

def computePiMethod1():
    e0 = time.time()
    Q = sp.dok_matrix((size,size)) 
    fillOffDiagonal(Q)
    # Set the diagonal of Q such that the row sums are zero
    Q.setdiag( -Q*ones(size) )
    # Compute a suitable stochastic matrix by means of uniformization
    l = min(Q.values())*1.001  # avoid periodicity, see trivedi's book
    P = sp.speye(size, size) - Q/l
    # compute Pi
    P =  P.tocsr()
    pi = zeros(size);  pi1 = zeros(size)
    pi[0] = 1;
    n = norm(pi - pi1,1); i = 0; 
    while n > 1e-3 and i < 1e5:
        pi1 = pi*P
        pi = pi1*P   # avoid copying pi1 to pi
        n = norm(pi - pi1,1); i += 1
    print "Method 1: ", time.time() - e0, i
    return pi

def computePiMethod2():
    e0 = time.time()
    Q = pysparse.spmatrix.ll_mat(size,size)
    fillOffDiagonal(Q)
    # fill diagonal
    x =  empty(size)
    Q.matvec(ones(size),x)
    Q.put(-x)
    # uniformize 
    l = min(Q.values())*1.001
    P = pysparse.spmatrix.ll_mat(size,size)
    P.put(ones(size))
    P.shift(-1./l, Q)
    # Compute pi
    P = P.to_csr()
    pi = zeros(size);  pi1 = zeros(size)
    pi[0] = 1;
    n = norm(pi - pi1,1); i = 0; 
    while n > 1e-3 and i < 1e5:
        P.matvec_transp(pi,pi1)
        P.matvec_transp(pi1,pi) 
        n = norm(pi - pi1,1); i += 1
    print "Method 2: ", time.time() - e0, i
    return pi

def plotPi(pi):
    pi = pi.reshape(N2,N1)
    matshow(pi)
    savefig("pi.png")

if __name__ == "__main__":
    pi = computePiMethod1()
    pi = computePiMethod2()
    plotPi(pi)
