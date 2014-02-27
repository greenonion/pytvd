"""
tvdip.py
~~~~~~~~

This module is a direct port of the original [1] tvdip Matlab script into
NumPy.

[1] M.A. Little, Nick S. Jones (2010) "Sparse Bayesian Step-Filtering for High-
Throughput Analysis of Molecular Machine Dynamics", in 2010 IEEE International
Conference on Acoustics, Speech and Signal Processing, 2010, ICASSP 2010
Proceedings.
"""

import numpy as np
import scipy as Sci
from scipy import sparse
from scipy.sparse import linalg
import sys


def tvdiplmax(y):
    """Calculate the value of lambda so that if lambda >= lambdamax, the TVD
    functional solved by TVDIP is minimized by the trivial constant solution
    x = mean(y). This can then be used to determine a useful range of values
    of lambda, for example.

    Args:
        y: Original signal to denoise, size N x 1.

    Returns:
        lambdamax: Value of lambda at which x = mean(y) is the output of the
            TVDIP function.

    """

    N = y.size
    M = N - 1

    # Construct sparse operator matrices
    I1 = sparse.eye(M)
    O1 = sparse.dia_matrix((M, 1))
    D = sparse.hstack([I1, O1]) - sparse.hstack([O1, I1])

    DDT = D.dot(D.conj().T)
    Dy = D.dot(y)

    lambdamax = np.absolute(linalg.spsolve(DDT, Dy)).max(0)

    return lambdamax


def tvdip(y, lambdas, display=1, stoptol=1e-3, maxiter=60):
    """Performs discrete total variation denoising (TVD) using a primal-dual
    interior-point solver. It minimizes the following discrete functional:

    E=(1/2)||y-x||_2^2+lambda*||Dx||_1

    over the variable x, given the input signal y, according to each value of
    the regularization parametero lambda > 0. D is the first difference matrix.
    Uses hot-restarts from each value of lambda to speed up convergence for
    subsequent values: best use of the feature is made by ensuring that the
    chosen lambda values are close to each other.

    Args:
        y: Original signal to denoise, size N x 1.
        lambdas: A vector of positive regularization parameters, size L x 1.
            TVD will be applied to each value in the vector.
        display: (Optional) Set to 0 to turn off progress display, 1 to turn
            on. Defaults to 1.
        stoptol: (Optional) Precision as determined by duality gap tolerance,
            if not specified defaults to 1e-3.
        maxiter: (Optional) Maximum interior-point iterations, if not specified
            defaults to 60.

    Returns:
        x: Denoised output signal for each value of lambda, size N x L.
        E: Objective functional at minimum for each lamvda, size L x 1.
        s: Optimization result, 1 = solved, 0 = maximum iterations
            exceeded before reaching duality gap tolerance, size L x 1.
        lambdamax: Maximum value of lambda for the given y. If
            lambda >= lambdamax, the output is the trivial constant solution
            x = mean(y).

    Example:
        >>> import numpy as np
        >>> import tvdip as tv
        >>> # Find the value of lambda greater than which the TVD solution is
        >>> # just the mean.
        >>> lmax = tv.tvdiplmax(y)
        >>> # Perform TV denoising for lambda across a range of values up to a
        >>> # small fraction of the maximum found above.
        >>> lratio = np.array([1e-4, 1e-3, 1e-2, 1e-1])
        >>> x, E, status, l_max = tv.tvdip(y, lmax*lratio, True, 1e-3)
        >>> plot(x[:,0])
    """

    # Search tuning parameters
    ALPHA = 0.01    # Backtracking linesearch parameter (0,0.5]
    BETA = 0.5      # Backtracking linesearch parameter (0,1)
    MAXLSITER = 20  # Max iterations of backtracking linesearch
    MU = 2          # t update

    N = y.size  # Length of input signal y
    M = N - 1   # Size of Dx

    # Construct sparse operator matrices
    I1 = sparse.eye(M)
    O1 = sparse.dia_matrix((M, 1))
    D = sparse.hstack([I1, O1]) - sparse.hstack([O1, I1])

    DDT = D.dot(D.conj().T)
    Dy = D.dot(y)

    # Find max value of lambda
    lambdamax = (np.absolute(linalg.spsolve(DDT, Dy))).max(0)

    if display:
        print "lambda_max=%5.2e" % lambdamax

    L = lambdas.size
    x = np.zeros((N, L))
    s = np.zeros((L, 1))
    E = np.zeros((L, 1))

    # Optimization variables set up once at the start
    z = np.zeros((M, 1))
    mu1 = np.ones((M, 1))
    mu2 = np.ones((M, 1))

    # Work through each value of lambda, with hot-restart on optimization
    # variables
    for idx, l in enumerate(lambdas):
        t = 1e-10
        step = np.inf
        f1 = z - l
        f2 = -z - l

        # Main optimization loop
        s[idx] = 1

        if display:
            print "Solving for lambda={0:5.2e}, lambda/lambda_max={1:5.2e}".format(l, l/lambdamax)
            print "Iter# primal    Dual    Gap"

        for iters in xrange(maxiter):
            DTz = (z.conj().T * D).conj().T
            DDTz = D.dot(DTz)
            w = Dy - (mu1 - mu2)

            # Calculate objectives and primal-dual gap
            pobj1 = 0.5*w.conj().T.dot(linalg.spsolve(DDT,w))+l*(np.sum(mu1+mu2))
            pobj2 = 0.5*DTz.conj().T.dot(DTz)+l*np.sum(np.absolute(Dy-DDTz))
            pobj = np.minimum(pobj1, pobj2)
            dobj = -0.5*DTz.conj().T.dot(DTz) + Dy.conj().T.dot(z)
            gap = pobj - dobj
            if display:
                print "{:5d} {:7.2e} {:7.2e} {:7.2e}".format(iters, pobj[0, 0],
                                                             dobj[0, 0],
                                                             gap[0, 0])

            # Test duality gap stopping criterion
            if gap <= stoptol:
                s[idx] = 1
                break

            if step >= 0.2:
                t = np.maximum(2*M*MU/gap, 1.2*t)

            # Do Newton step
            rz = DDTz - w
            Sdata = (mu1/f1 + mu2/f2)
            S = DDT-sparse.csc_matrix((Sdata.reshape(Sdata.size),
                                      (np.arange(M), np.arange(M))))
            r = -DDTz + Dy + (1/t)/f1 - (1/t)/f2
            dz = linalg.spsolve(S, r).reshape(r.size, 1)
            dmu1 = -(mu1+((1/t)+dz*mu1)/f1)
            dmu2 = -(mu2+((1/t)-dz*mu2)/f2)

            resDual = rz.copy()
            resCent = np.vstack((-mu1*f1-1/t, -mu2*f2-1/t))
            residual = np.vstack((resDual, resCent))

            # Perform backtracking linesearch
            negIdx1 = dmu1 < 0
            negIdx2 = dmu2 < 0
            step = 1
            if np.any(negIdx1):
                step = np.minimum(step,
                                  0.99*(-mu1[negIdx1]/dmu1[negIdx1]).min(0))
            if np.any(negIdx2):
                step = np.minimum(step,
                                  0.99*(-mu2[negIdx2]/dmu2[negIdx2]).min(0))

            for _ in xrange(MAXLSITER):
                newz = z + step*dz
                newmu1 = mu1 + step*dmu1
                newmu2 = mu2 + step*dmu2
                newf1 = newz - l
                newf2 = -newz - l

                # Update residuals
                newResDual = DDT.dot(newz) - Dy + newmu1 - newmu2
                newResCent = np.vstack((-newmu1*newf1-1/t, -newmu2*newf2-1/t))
                newResidual = np.vstack((newResDual, newResCent))

                if (np.maximum(newf1.max(0), newf2.max(0)) < 0
                    and (Sci.linalg.norm(newResidual) <=
                        (1-ALPHA*step)*Sci.linalg.norm(residual))):
                    break

                step = BETA * step

            # Update primal and dual optimization parameters
            z = newz
            mu1 = newmu1
            mu2 = newmu2
            f1 = newf1
            f2 = newf2

        x[:, idx] = (y-D.conj().T.dot(z)).reshape(x.shape[0])
        xval = x[:, idx].reshape(x.shape[0], 1)
        E[idx] = 0.5*np.sum((y-xval)**2)+l*np.sum(np.absolute(D.dot(xval)))

        # We may have a close solution that does not satisfy the duality gap
        if iters >= maxiter:
            s[idx] = 0

        if display:
            if s[idx]:
                print("Solved to precision of duality gap %5.2e") % gap
            else:
                print("Max iterations exceeded - solution may be inaccurate")

    return x, E, s, lambdamax
