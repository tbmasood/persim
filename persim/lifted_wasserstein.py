"""
    Modification of Wasserstein distance avilable in persim  to implement the 
    Lifted Wasserstein distance as described in: Soler et al. "Lifted Wasserstein 
    Matcher for Fast and Robust Topology Tracking"
    Author: Talha Bin Masood
"""
import numpy as np
from sklearn import metrics
from scipy import optimize
import warnings
import math

__all__ = ["lifted_wasserstein"]


def lifted_wasserstein(dgm1, dgm2, alpha=1.0, beta=1.0, gamma=1.0, order=1.0, matching=False):
    """
    Perform the lifted Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, and the next three columns contain the (x, y, z) spatial co-ordinates
    of the extrema.
    
    See the `distances` notebook for an example of how to use this.
    Parameters
    ------------
    dgm1: Mx(>=8) 
        PD 1 specified as an array of birth/death pairs followed by (x,y,z) 
        locations of extrema and (x,y,z) coordinates of the saddle point.
    dgm2: Nx(>8) 
        PD 2 specified as an array of birth/death pairs followed by (x,y,z) 
        locations of extrema and (x,y,z) coordinates of the saddle point.
    alpha: float, default 1.0
        weight factor for birth column
    beta: float, default 1.0
        weight factor for death column
    gamma: float, default 1.0
        weight factor for the x,y,z columns
    order: float, default 1.0
        the parameter q in q-Wasserstein distance
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
        S = np.array([[0, 0, 0, 0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0, 0, 0, 0]])
        N = 1
    
    def compute_dist(del_b, del_d, del_x, del_y, del_z): 
        if order == 1.0:
            return alpha * del_b + beta * del_d + gamma*(del_x + del_y + del_z)
        else:
            return (alpha * del_b**order + beta * del_d**order \
                + gamma*(del_x**order + del_y**order + del_z**order))**(1.0/order) 
    
    # Step 1: Compute CSM between S and T, including points on diagonal
    D = np.zeros((M+N, M+N))
    for i in range(M):
        for j in range(N):
            del_b = abs(S[i, 0] - T[j, 0])
            del_d = abs(S[i, 1] - T[j, 1])
            del_x = abs(S[i, 2] - T[j, 2])
            del_y = abs(S[i, 3] - T[j, 3])
            del_z = abs(S[i, 4] - T[j, 4])
            D[i, j] = D[j, i] = compute_dist(del_b, del_d, del_x, del_y, del_z)
            
    UR = np.max(D)*np.ones((M, M))
    for i in range(M):
        del_b = abs(S[i, 0])
        del_d = abs(S[i, 1])
        del_x = abs(S[i, 2] - S[i, 5])
        del_y = abs(S[i, 3] - S[i, 6])
        del_z = abs(S[i, 4] - S[i, 7])
        UR[i][i] = compute_dist(del_b, del_d, del_x, del_y, del_z)
        
    D[0:M, N:N+M] = UR
    UL = np.max(D)*np.ones((N, N))
    for i in range(N):
        del_b = abs(T[i, 0])
        del_d = abs(T[i, 1])
        del_x = abs(T[i, 2] - T[i, 5])
        del_y = abs(T[i, 3] - T[i, 6])
        del_z = abs(T[i, 4] - T[i, 7])
        UL[i][i] = compute_dist(del_b, del_d, del_x, del_y, del_z)
            
    D[M:N+M, 0:N] = UL

    # Step 2: Run the hungarian algorithm
    matchi, matchj = optimize.linear_sum_assignment(D)
    if order == 1.0:
        matchdist = np.sum(D[matchi, matchj])
    elif order == math.inf:
        # Taking max value here provides the L_\inf distance 
        matchdist = np.max(D[matchi, matchj])
    else:
        matchdist = np.sum(D[matchi, matchj]**order)**(1.0/order)

    if matching:
        matchidx = [(i, j) for i, j in zip(matchi, matchj)]
        return matchdist, (matchidx, D)

    return matchdist
