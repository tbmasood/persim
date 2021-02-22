"""
    Implementation of the Wasserstein distance using
    the Hungarian algorithm
    Author: Chris Tralie
"""
import numpy as np
from sklearn import metrics
from scipy import optimize
import warnings
import math

__all__ = ["wasserstein"]


def wasserstein(dgm1, dgm2, order=1.0, internal_p=math.inf, matching=False):
    """
    Perform the Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).
    See the `distances` notebook for an example of how to use this.
    Parameters
    ------------
    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching information and cross-similarity matrix
    Returns 
    ---------
    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)
    """

    S = np.array(dgm1)
    M = min(S.shape[0], S.size)
    if S.size > 0:
        S = S[np.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            warnings.warn(
                "dgm1 has points with non-finite death times;"+
                "ignoring those points"
            )
            M = S.shape[0]
    T = np.array(dgm2)
    N = min(T.shape[0], T.size)
    if T.size > 0:
        T = T[np.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            warnings.warn(
                "dgm2 has points with non-finite death times;"+
                "ignoring those points"
            )
            N = T.shape[0]

    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1
    
    # Step 1: Compute CSM between S and dgm2, including points on diagonal

    if internal_p == 1.0:
        DUL = metrics.pairwise.pairwise_distances(S, T, metric = "l1")
    elif internal_p == 2.0:
        DUL = metrics.pairwise.pairwise_distances(S, T)
    elif internal_p == math.inf:
        DUL = metrics.pairwise.pairwise_distances(S, T, metric = "chebyshev") 
    else:
        DUL = metrics.pairwise.pairwise_distances(S, T, metric = "minkowski", p=internal_p) 
    
    # Put diagonal elements into the matrix
    # Rotate the diagrams to make it easy to find the straight line
    # distance to the diagonal
    cp = np.cos(np.pi/4)
    sp = np.sin(np.pi/4)
    R = np.array([[cp, -sp], [sp, cp]])
    S = S[:, 0:2].dot(R)
    T = T[:, 0:2].dot(R)
    D = np.zeros((M+N, M+N))
    D[0:M, 0:N] = DUL
    UR = np.max(D)*np.ones((M, M))
    np.fill_diagonal(UR, S[:, 1])
    D[0:M, N:N+M] = UR
    UL = np.max(D)*np.ones((N, N))
    np.fill_diagonal(UL, T[:, 1])
    D[M:N+M, 0:N] = UL

    # Step 2: Run the hungarian algorithm
    matchi, matchj = optimize.linear_sum_assignment(D)
    if order == 1.0:
        matchdist = np.sum(D[matchi, matchj])
    elif order == math.inf:
        matchdist = np.max(D[matchi, matchj])
    else:
        matchdist = np.sum(D[matchi, matchj]**order)**(1.0/order)

    if matching:
        matchidx = [(i, j) for i, j in zip(matchi, matchj)]
        return matchdist, (matchidx, D)

    return matchdist
