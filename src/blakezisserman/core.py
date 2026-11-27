
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import norm, cg, LinearOperator

import numpy as np
from numpy import ones, zeros, sum, any, min, sqrt, inf, diff

import logging
# set logger
log = logging.getLogger('bz')
log.setLevel(logging.WARNING)
# console output handler
formatter = logging.Formatter("%(levelname)8.8s; %(asctime)s; %(message)s", datefmt='%d-%m-%Y %H:%M:%S')
logsh = logging.StreamHandler()
logsh.setFormatter(formatter)
log.addHandler(logsh)





# Optimize the 1-dimensional (curves) Blake-Zisserman functional model 
def solve1d(
    values,
    breaks,
    llambda,
    eta, 
    epsilon, 
    bound_cond,
    fun_rtol, 
    maxiter, 
    pcg_maxiter_z, 
    pcg_maxiter_u,
    log_level
):

    """
      Block coordinate descent algorithm for the minimization of the Blake-Zisserman relaxed functional for N-dimensional curves.

      Output:
      - u           Numpy 2d-array of the input approximation
      - z           Numpy 1d-array of the gradient discontinuity of the approximation
      - t           Numpy 1d-array of the break points of the curve parametrization
      - F           Numpy 1d-array of functional evaluations during minimization iteration
      - converged   Boolean flag for convergence of the functional optimization

    """  

    log.setLevel(log_level)

    log.info(f"Launched solve1d with parameters: llambda={llambda:.0e}, eta={eta:.0e}, epsilon={epsilon:.0e}, bound_cond={bound_cond}, fun_rtol={fun_rtol:.0e}, maxiter={maxiter}, pcg_maxiter_z={pcg_maxiter_z}, pcg_maxiter_u={pcg_maxiter_u}")

    #########
    # SETUP #
    #########

    g = values
    # P = number of points of curve parametrization
    # N = dimension of the space the curve is embedded in
    P, N = g.shape

    # Discretization of the curve domain (from break points of the curve parametrization)
    if breaks is None:
        t = np.arange(0, P)
        h = ones(P)
    else:
        # Get intervals length by forward difference of the break points
        t = breaks
        h = diff(t, n=1, axis=0)
        h = np.append(h, h[-1])    # P-th interval is a backward difference

    # Differential operators (1st-order forward, 2nd-order central)
    Dt  = spdiags([-ones(P), ones(P)], [0,1], P, P).tolil()
    Dtt = spdiags([ones(P), -2*ones(P), ones(P)], [-1,0,1], P, P).tolil()
    # Boundary conditions entries editing, allowed by .tolil() 
    if bound_cond == 'none':
        pass
    elif bound_cond == 'periodic':
        Dt[P-1, 0]  = 1
        Dtt[P-1, 0] = 1
        Dtt[0, P-1] = 1
    elif bound_cond == 'null-neumann':
        Dt[P-1, P-1]  = 0
        Dtt[0, 0]     = -1
        Dtt[P-1, P-1] = -1
    # Apply discretization lengths
    Dt  = spdiags(1/h, 0, P, P) @ Dt
    Dtt = spdiags(1/(h**2), 0, P, P) @ Dtt
    # Restore efficient sparse format
    Dt   = Dt.tocsr()
    Dtt  = Dtt.tocsr()
    DttT = Dtt.T.tocsr()
    Dt2  = (Dt.T@Dt).tocsr()

    # Constant values
    I    = eye(P)
    E    = ones(P)
    b_z  = eta/(2*epsilon)*E
    b_u  = 2*llambda*g

    # Initial values
    z = 1.*E
    u = 1.*g
    d_z = zeros(P)
    d_u = zeros((P, N))

    # Functional evaluation
    def Fe(z, u):
        du  = Dtt@u
        dz  = Dt@z
        res = u-g
        return (
            np.sum(z[:, None]**2 * du**2) + llambda*np.sum(res**2) + eta*(epsilon*np.dot(dz, dz)+np.dot(z-E, z-E)/(4*epsilon))
        )

    # Energy values
    F = zeros(maxiter + 1)
    F_0 = Fe(z, u)
    F[0] = F_0

    ########################
    # ENERGY OPTIMIZATION  #
    ########################

    converged = False

    for k in range(1, maxiter + 1):

        log.info(f'iter={k}')

        # Solve for variable z

        A_z = 2*spdiags(sum((Dtt@u)**2, axis=1), 0, P, P) + 2*epsilon*eta*Dt2 + eta/(2*epsilon)*I
        
        # Preconditioning and tolerance for PCG
        Ai_z = 1. / A_z.diagonal()
        M_z  = LinearOperator(A_z.shape, matvec=lambda x: Ai_z*x, dtype=A_z.dtype)
        pcg_tol_z = sqrt(min(A_z.diagonal()-2*epsilon*eta*4)/norm(A_z, inf))

        # Gradient
        grad_z = A_z@z - b_z

        if any(grad_z):
            d_z, exit_z = cg(A_z, -grad_z, x0=d_z, rtol=pcg_tol_z, maxiter=pcg_maxiter_z, M=M_z)
            temp_d_z = d_z.T@grad_z
            den_z    = d_z.T@A_z@d_z
            step_z   = -temp_d_z/den_z
            z        += step_z*d_z
            
            if exit_z==0:
                log.info(f"Internal PCG in z: succesfully converged")
            else:
                log.warning(f"Internal PCG in z: failed to converge")
        
        # Solve for variable u

        z2Dtt = Dtt.multiply((z**2)[:, None]).tocsr()
        A_u = 2 * (DttT @ z2Dtt) + 2*llambda*I

        # Preconditioning and tolerance for PCG
        Ai_u = 1. / A_u.diagonal()
        M_u  = LinearOperator(A_u.shape, matvec=lambda x: Ai_u*x, dtype=A_u.dtype)
        pcg_tol_u = sqrt(2*llambda/norm(A_u, inf))
        
        for n in range(0, N):

            # Gradient
            grad_u = A_u@u[:, n] - b_u[:, n]
            
            if any(grad_u):
                d_u[:, n], exit_u = cg(A_u, -grad_u, x0=d_u[:,n], rtol=pcg_tol_u, maxiter=pcg_maxiter_u, M=M_u)
                temp_d_u = d_u[:, n].T@grad_u
                den_u    = d_u[:, n].T@A_u@d_u[:, n]
                step_u   = -1.5*temp_d_u/den_u
                u[:, n]  += step_u*d_u[:, n]
                
                if exit_u==0:
                    log.info(f"Internal PCG in u{n+1}: succesfully converged")
                else:
                    log.warning(f"Internal PCG in u{n+1}: failed to converge")

        # Convergence criterion of relative variation
        F_1 = Fe(z, u)
        fun_rvar = abs(F_1-F_0) / abs(F_0)
        F[k] = F_1
        F_0 = F_1

        if fun_rvar < fun_rtol:
            converged = True
            break

    if converged:
        log.info(f'Successfully converged at iter={k} within functional tolerance fun_rtol={fun_rtol}')
        # Close the loop for closed curves
        if bound_cond=='periodic' and N>1:
            u[-1] = u[0]
            z[-1] = z[0]
    else:
        log.warning(f'Failed to converge at maxiter={maxiter} within functional tolerance fun_rtol={fun_rtol}')

    return u, z, t, F, converged










#############################################################################################
# Fast linear algebra for solve2d                                                          #
#                                                                                           #
# Nearly all of solve2d's run time is the Jacobi-PCG solve for u (hundreds of iterations   #
# per outer iteration). This section keeps that algorithm and makes each iteration cheaper: #
#   - one PCG for all N channels at once, so the matrix is read once per iteration;         #
#   - matrices stored by diagonals (the operators are fixed stencils): no column indices,   #
#     one third less memory traffic than CSR;                                               #
#   - the PCG vector operations fused into 3 passes, run in parallel with numba             #
#     (pure numpy fallback if numba is missing);                                            #
#   - the u matrix values computed by a precomputed linear map of (z^2, (s+eps^2)^2),       #
#     instead of sparse-sparse products at every outer iteration.                           #
# Reductions are summed per fixed-size block, so results do not depend on the thread count.#
#############################################################################################

from scipy.sparse import coo_matrix

_BLK = 4096

try:
    import numba as _nb
    import os as _os
    # Prefer OpenMP over TBB: avoids numba probing (and warning about) an outdated TBB.
    # An explicit NUMBA_THREADING_LAYER / NUMBA_THREADING_LAYER_PRIORITY set by the user wins.
    if not _os.environ.get('NUMBA_THREADING_LAYER') and not _os.environ.get('NUMBA_THREADING_LAYER_PRIORITY'):
        _nb.config.THREADING_LAYER_PRIORITY = ['omp', 'workqueue', 'tbb']

    @_nb.njit(parallel=True, cache=True)
    def _nb_dia_matmat(data, offs, H, Xp, Y, partial, want_dot):
        # Y = A @ X, with A stored by diagonals and X zero-padded by H rows on each side.
        # If want_dot: partial[b, c] = sum over block b of X[:, c]*Y[:, c]
        n, K = data.shape
        k = Y.shape[1]
        nblk = (n + _BLK - 1) // _BLK
        for b in _nb.prange(nblk):
            lo = b * _BLK
            hi = lo + _BLK
            if hi > n:
                hi = n
            if want_dot:
                for c in range(k):
                    partial[b, c] = 0.0
            if k == 3:      # fast path: RGB, accumulators in registers
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                for i in range(lo, hi):
                    y0 = 0.0
                    y1 = 0.0
                    y2 = 0.0
                    for kk in range(K):
                        a = data[i, kk]
                        j = i + H + offs[kk]
                        y0 += a * Xp[j, 0]
                        y1 += a * Xp[j, 1]
                        y2 += a * Xp[j, 2]
                    Y[i, 0] = y0
                    Y[i, 1] = y1
                    Y[i, 2] = y2
                    if want_dot:
                        s0 += Xp[i + H, 0] * y0
                        s1 += Xp[i + H, 1] * y1
                        s2 += Xp[i + H, 2] * y2
                if want_dot:
                    partial[b, 0] = s0
                    partial[b, 1] = s1
                    partial[b, 2] = s2
            elif k == 1:    # fast path: grayscale and the s, z systems
                s0 = 0.0
                for i in range(lo, hi):
                    y0 = 0.0
                    for kk in range(K):
                        y0 += data[i, kk] * Xp[i + H + offs[kk], 0]
                    Y[i, 0] = y0
                    if want_dot:
                        s0 += Xp[i + H, 0] * y0
                if want_dot:
                    partial[b, 0] = s0
            else:
                for i in range(lo, hi):
                    for c in range(k):
                        Y[i, c] = 0.0
                    for kk in range(K):
                        a = data[i, kk]
                        j = i + H + offs[kk]
                        for c in range(k):
                            Y[i, c] += a * Xp[j, c]
                    if want_dot:
                        for c in range(k):
                            partial[b, c] += Xp[i + H, c] * Y[i, c]

    @_nb.njit(parallel=True, cache=True)
    def _nb_pupdate(P, R, Minv, beta):
        # P = beta*P + Minv*R   (same rounding as scipy's  p *= beta; p += z)
        n, k = P.shape
        for i in _nb.prange(n):
            mi = Minv[i]
            for c in range(k):
                P[i, c] = beta[c] * P[i, c] + mi * R[i, c]

    @_nb.njit(parallel=True, cache=True)
    def _nb_update(X, R, P, Q, alpha, Minv, prr, prz):
        # X += alpha*P; R -= alpha*Q; block partial sums of r.r and r.(M^-1 r)
        n, k = P.shape
        nblk = prr.shape[0]
        for b in _nb.prange(nblk):
            lo = b * _BLK
            hi = lo + _BLK
            if hi > n:
                hi = n
            if k == 3:      # fast path: RGB, accumulators in registers
                a0 = alpha[0]
                a1 = alpha[1]
                a2 = alpha[2]
                r0 = 0.0
                r1 = 0.0
                r2 = 0.0
                z0 = 0.0
                z1 = 0.0
                z2 = 0.0
                for i in range(lo, hi):
                    mi = Minv[i]
                    X[i, 0] += a0 * P[i, 0]
                    X[i, 1] += a1 * P[i, 1]
                    X[i, 2] += a2 * P[i, 2]
                    t0 = R[i, 0] - a0 * Q[i, 0]
                    t1 = R[i, 1] - a1 * Q[i, 1]
                    t2 = R[i, 2] - a2 * Q[i, 2]
                    R[i, 0] = t0
                    R[i, 1] = t1
                    R[i, 2] = t2
                    r0 += t0 * t0
                    r1 += t1 * t1
                    r2 += t2 * t2
                    z0 += t0 * (mi * t0)
                    z1 += t1 * (mi * t1)
                    z2 += t2 * (mi * t2)
                prr[b, 0] = r0
                prr[b, 1] = r1
                prr[b, 2] = r2
                prz[b, 0] = z0
                prz[b, 1] = z1
                prz[b, 2] = z2
            else:
                for c in range(k):
                    prr[b, c] = 0.0
                    prz[b, c] = 0.0
                for i in range(lo, hi):
                    mi = Minv[i]
                    for c in range(k):
                        al = alpha[c]
                        X[i, c] += al * P[i, c]
                        r = R[i, c] - al * Q[i, c]
                        R[i, c] = r
                        prr[b, c] += r * r
                        prz[b, c] += r * (mi * r)

    _HAVE_NUMBA = True

except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False


class _DiaMatrix:
    """
    Square matrix with a fixed set of diagonals, stored row-wise: A[i, i+offs[k]] = data[i, k].
    Its values are an affine function of per-pixel weights:  data = sum_g W_g @ w_g + const.
    """

    def __init__(self, terms, n, diag_const):
        """
        A(w) = sum_t c_t * D_t^T diag(w_{g_t}) D_t + diag_const*Id,
        terms: list of (D csr, c_t, g_t).
        """
        pairs = [self._pairs(D) for D, _, _ in terms]
        offs = np.unique(np.concatenate([np.unique(j - i) for pp in pairs for i, j, _, _ in pp] + [np.zeros(1, np.int64)]))
        K = offs.size
        self.n, self.K, self.offs = n, K, offs.astype(np.int64)
        self.H = int(np.abs(offs).max())
        self.k0 = int(np.searchsorted(offs, 0))
        ngroups = 1 + max(g for _, _, g in terms)
        self.maps = []
        for g in range(ngroups):
            P_, K_, V_ = [], [], []
            for (D, coef, gg), pp in zip(terms, pairs):
                if gg != g:
                    continue
                for i, j, k, v in pp:
                    P_.append(i * K + np.searchsorted(offs, j - i)); K_.append(k); V_.append(coef * v)
            self.maps.append(coo_matrix((np.concatenate(V_), (np.concatenate(P_), np.concatenate(K_))),
                                        shape=(n * K, n)).tocsr())
        self.const = np.zeros((n, K)); self.const[:, self.k0] = diag_const
        self.data = np.zeros((n, K))

    @staticmethod
    def _pairs(D):
        # All products D[k,i]*D[k,j] over the rows k of D (D has a few nonzeros per row)
        D = D.tocsr(); D.sort_indices()
        cnt = np.diff(D.indptr); m = cnt.max(); rows = np.arange(D.shape[0])
        out = []
        for a in range(m):
            for b in range(m):
                k = rows[cnt > max(a, b)]
                p = D.indptr[k] + a; q = D.indptr[k] + b
                out.append((D.indices[p].astype(np.int64), D.indices[q].astype(np.int64), k, D.data[p] * D.data[q]))
        return out

    def set_weights(self, *w):
        data = self.const.copy().ravel()
        for W, wg in zip(self.maps, w):
            data += W @ wg
        self.data = data.reshape(self.n, self.K)
        return self.diagonal(), self.norm_inf()

    def add_diagonal(self, base_data, d):
        self.data = base_data.copy()
        self.data[:, self.k0] += d
        return self.diagonal(), self.norm_inf()

    def diagonal(self):
        return self.data[:, self.k0].copy()

    def norm_inf(self):
        return np.abs(self.data).sum(axis=1).max()

    def padded(self, X):
        Xp = np.zeros((self.n + 2*self.H, X.shape[1]))
        Xp[self.H:self.H + self.n] = X
        return Xp

    def matmat(self, X, Xp=None, Y=None, partial=None):
        """Y = A @ X (X dense (n, k)). Pass a padded Xp to skip the copy; returns (Y, X.Y per column)."""
        if Xp is None:
            Xp = self.padded(np.asarray(X, dtype=np.float64))
        k = Xp.shape[1]
        if Y is None:
            Y = np.empty((self.n, k))
        if _HAVE_NUMBA:
            want = partial is not None
            if partial is None:
                partial = np.empty((1, k))
            _nb_dia_matmat(self.data, self.offs, self.H, Xp, Y, partial, want)
            return Y, (partial.sum(axis=0) if want else None)
        Y[:] = 0.0
        for kk in range(self.K):
            o = self.H + self.offs[kk]
            Y += self.data[:, kk:kk+1] * Xp[o:o + self.n]
        return Y, (np.einsum('ij,ij->j', Xp[self.H:self.H + self.n], Y) if partial is not None else None)


def _block_pcg(A, Minv, B, X0, rtol, maxiter):
    """
    Jacobi-preconditioned conjugate gradient on the columns of B simultaneously.

    Column by column this is the same algorithm (stopping rule ||r|| < rtol*||b||, return
    values, warm start, rounding of the vector updates) as
    scipy.sparse.linalg.cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter, M=diag(Minv)).
    Returns X (n, k) and exit codes (k,). Converged columns are dropped from the working set.
    """
    n, k = B.shape
    B = np.ascontiguousarray(B, dtype=np.float64)
    Xout = np.array(X0, dtype=np.float64, order='C', copy=True)
    info = np.zeros(k, dtype=np.int64)

    bnrm2 = np.sqrt(np.einsum('ij,ij->j', B, B))
    zero = bnrm2 == 0
    Xout[:, zero] = B[:, zero]              # scipy returns b when ||b|| == 0
    cols = np.flatnonzero(~zero)            # original indices of the working columns
    if cols.size == 0:
        return Xout, info
    atol = rtol * bnrm2[cols]

    H = A.H
    nblk = (n + _BLK - 1) // _BLK

    def alloc(kk):
        Pp = np.zeros((n + 2*H, kk))        # search direction, zero padded for the stencil
        return Pp, Pp[H:H + n], np.empty((n, kk)), np.empty((nblk, kk)), np.empty((nblk, kk)), np.empty((nblk, kk))

    X = np.ascontiguousarray(Xout[:, cols])
    R = np.ascontiguousarray(B[:, cols]) - A.matmat(X)[0]
    rr = np.einsum('ij,ij->j', R, R)
    rz = np.einsum('ij,ij->j', R, Minv[:, None] * R)
    Pp, P, Q, p1, p2, p3 = alloc(cols.size)
    rho_prev = np.ones(cols.size)

    for it in range(maxiter):
        done = np.sqrt(rr) < atol
        if done.any():
            Xout[:, cols[done]] = X[:, done]
            keep = ~done
            if not keep.any():
                return Xout, info
            Pold = P[:, keep]
            cols, atol, rr, rz, rho_prev = cols[keep], atol[keep], rr[keep], rz[keep], rho_prev[keep]
            X, R = np.ascontiguousarray(X[:, keep]), np.ascontiguousarray(R[:, keep])
            Pp, P, Q, p1, p2, p3 = alloc(cols.size)
            P[:] = Pold
        beta = rz / rho_prev if it > 0 else np.zeros(cols.size)
        if _HAVE_NUMBA:
            _nb_pupdate(P, R, Minv, beta)
            _, pq = A.matmat(None, Xp=Pp, Y=Q, partial=p1)
            alpha = rz / pq
            rho_prev = rz
            _nb_update(X, R, P, Q, alpha, Minv, p2, p3)
            rr = p2.sum(axis=0)
            rz = p3.sum(axis=0)
        else:
            P *= beta; P += Minv[:, None] * R
            _, pq = A.matmat(None, Xp=Pp, Y=Q, partial=p1)
            alpha = rz / pq
            rho_prev = rz
            X += alpha * P
            R -= alpha * Q
            rr = np.einsum('ij,ij->j', R, R)
            rz = np.einsum('ij,ij->j', R, Minv[:, None] * R)

    # for loop exhausted: return incomplete progress
    Xout[:, cols] = X
    info[cols] = maxiter
    return Xout, info



# Optimize the 2-dimensional (images) Blake-Zisserman functional model
def solve2d(
    values,
    mu,
    alpha,
    beta,
    delta,
    epsilon,
    fun_rtol,
    maxiter,
    pcg_maxiter_s,
    pcg_maxiter_z,
    pcg_maxiter_u,
    log_level
):

    """
      Block coordinate descent algorithm for the minimization of the Blake-Zisserman relaxed functional for N-dimensional images.

      Output:
      - u           Numpy nd-array of the input approximation
      - s           Numpy 2d-array of the discontinuity of the approximation
      - z           Numpy 2d-array of the gradient discontinuity of the approximation
      - F           Numpy nd-array of functional evaluations during minimization iteration
      - converged   Boolean flag for convergence of the functional optimization

    """

    log.setLevel(log_level)

    log.info(f"Launched solve2d with parameters: mu={mu:.0e}, alpha={alpha:.0e}, beta={beta:.0e}, delta={delta:.0e}, epsilon={epsilon:.0e}, fun_rtol={fun_rtol:.0e}, maxiter={maxiter}, pcg_maxiter_s={pcg_maxiter_s}, pcg_maxiter_z={pcg_maxiter_z}, pcg_maxiter_u={pcg_maxiter_u}")

    #########
    # SETUP #
    #########

    g = values

    # I = number of image rows
    # J = number of image columns
    # N = dimension of the space the image is embedded in
    I, J, N = g.shape
    n = I*J

    # Vectorize input by reshaping
    g = np.reshape(g, (n, N), order="F")

    # Differential operators (1st-order forward, 2nd-order central)
    Dx  = kron(eye(J), spdiags([-ones(I), ones(I)], [0,1], I, I), format="csr")
    Dy  = kron(spdiags([-ones(J), ones(J)], [0,1], J, J), eye(I), format="csr")
    Dxx = kron(eye(J), spdiags([ones(I), -2*ones(I), ones(I)], [-1,0,1], I, I), format="csr")
    Dyy = kron(spdiags([ones(J), -2*ones(J), ones(J)], [-1,0,1], J, J), eye(I), format="csr")
    Dxy = (Dy@Dx).tocsr()

    # Constant values
    E    = ones(n)
    b_s  = (alpha-beta)/(2*epsilon)*E
    b_z  = beta/(2*epsilon)*E
    b_u  = 2*mu*g
    xe   = epsilon**.25
    oe   = epsilon**2

    # s and z systems:  A_s = 2eps(alpha-beta) Dfix + ((alpha-beta)/(2eps) + 2 xe q) Id
    #                   A_z = 2eps beta Dfix       + (beta/(2eps) + 2 delta r) Id
    # with Dfix = Dx^T Dx + Dy^T Dy (fixed pattern, only the diagonal changes)
    Ms = _DiaMatrix([(Dx, 1., 0), (Dy, 1., 0)], n, 0.)
    Mz = _DiaMatrix([(Dx, 1., 0), (Dy, 1., 0)], n, 0.)
    Ms.set_weights(E)
    Dfix_data = Ms.data.copy()
    C_s = 2*epsilon*(alpha-beta)*Dfix_data
    C_z = 2*epsilon*beta*Dfix_data

    # u system: A_u = 2delta (Dxx^T Z Dxx + Dyy^T Z Dyy + 2 Dxy^T Z Dxy) + 2xe (Dx^T S Dx + Dy^T S Dy) + 2mu Id
    # with Z = diag(z^2), S = diag(s^2+eps^2); its values are a linear map of (z^2, s^2+eps^2)
    Mu = _DiaMatrix([(Dxx, 2*delta, 0), (Dyy, 2*delta, 0), (Dxy, 4*delta, 0), (Dx, 2*xe, 1), (Dy, 2*xe, 1)],
                    n, 2*mu)

    # Initial values
    s = 1.*E
    z = 1.*E
    u = 1.*g
    d_s = zeros((n, 1))
    d_z = zeros((n, 1))
    d_u = zeros((n, N))


    def derivative_energies(u):
        """Channel-summed derivative energies, reused until u changes."""
        dxu, dyu = Dx @ u, Dy @ u
        q = np.sum(dxu**2 + dyu**2, axis=1)
        dxxu, dyyu, dxyu = Dxx @ u, Dyy @ u, Dxy @ u
        r = np.sum(dxxu**2 + dyyu**2 + 2*dxyu**2, axis=1)
        return q, r

    # Functional evaluation using cached derivative energies.
    def Fe(s, z, u, q, r):
        dxs  = Dx@s
        dys  = Dy@s
        dxz  = Dx@z
        dyz  = Dy@z
        res = u-g
        return (
            delta * np.dot(z**2, r) +
            xe * np.dot(s**2+oe, q) +
            (alpha-beta) * (epsilon*(np.dot(dxs, dxs)+np.dot(dys, dys)) + np.dot(s-E, s-E)/(4*epsilon)) +
            beta *         (epsilon*(np.dot(dxz, dxz)+np.dot(dyz, dyz)) + np.dot(z-E, z-E)/(4*epsilon)) +
            mu*np.sum(res**2)
        )

    # One scalar block-coordinate step: PCG direction + exact line search
    def scalar_step(M, base, extra_diag, rhs, x, d, maxit, tol_shift, name, relax=1.):
        diag, nrm = M.add_diagonal(base, extra_diag)
        Minv = 1. / diag
        pcg_tol = sqrt(min(diag - tol_shift)/nrm)
        grad = M.matmat(x[:, None])[0][:, 0] - rhs
        if any(grad):
            d, exit_ = _block_pcg(M, Minv, -grad[:, None], d, pcg_tol, maxit)
            dv = d[:, 0]
            step = -relax*(dv@grad)/(dv@M.matmat(d)[0][:, 0])
            x += step*dv
            if exit_[0] == 0:
                log.info(f"Internal PCG in {name}: succesfully converged")
            else:
                log.warning(f"Internal PCG in {name}: failed to converge")
        return x, d

    # Energy values
    F = zeros(maxiter + 1)
    q, r = derivative_energies(u)
    F_0 = Fe(s, z, u, q, r)
    F[0] = F_0

    ########################
    # ENERGY OPTIMIZATION  #
    ########################

    converged = False

    for k in range(1, maxiter + 1):

        log.info(f'iter={k}')

        # Solve for variable s
        s, d_s = scalar_step(Ms, C_s, (alpha-beta)/(2*epsilon) + 2*xe*q, b_s, s, d_s, pcg_maxiter_s, 2*epsilon*(alpha-beta)*4, "s")

        # Solve for variable z
        z, d_z = scalar_step(Mz, C_z, beta/(2*epsilon) + 2*delta*r, b_z, z, d_z, pcg_maxiter_z, 2*epsilon*beta*4, "z")

        # Solve for variable u (all channels in one block PCG)
        diag_u, nrm_u = Mu.set_weights(z**2, s**2+oe)
        Minv_u = 1. / diag_u
        pcg_tol_u = sqrt(2*mu/nrm_u)

        grad_u = Mu.matmat(u)[0] - b_u
        cols = np.flatnonzero(np.any(grad_u != 0, axis=0))
        if cols.size:
            Dsol, exit_u = _block_pcg(Mu, Minv_u, -grad_u[:, cols], d_u[:, cols], pcg_tol_u, pcg_maxiter_u)
            d_u[:, cols] = Dsol
            AD = Mu.matmat(Dsol)[0]
            temp_d_u = np.einsum('ij,ij->j', Dsol, grad_u[:, cols])
            den_u    = np.einsum('ij,ij->j', Dsol, AD)
            step_u   = -1.5*temp_d_u/den_u
            u[:, cols] += step_u*Dsol

            if np.all(exit_u == 0):
                log.info("Internal PCG in u: succesfully converged")
            else:
                log.warning("Internal PCG in u: failed to converge")

        # Convergence criterion of relative variation
        q, r = derivative_energies(u)
        F_1 = Fe(s, z, u, q, r)
        fun_rvar = abs(F_1-F_0) / abs(F_0)
        F[k] = F_1
        F_0 = F_1

        if fun_rvar < fun_rtol:
            converged = True
            break

    if converged:
        log.info(f'Successfully converged at iter={k} within functional tolerance fun_rtol={fun_rtol}')
    else:
        log.warning(f'Failed to converge at maxiter={maxiter} within functional tolerance fun_rtol={fun_rtol}')

    return (
        u.reshape(I, J, N, order="F"),
        s.reshape(I, J, order="F"),
        z.reshape(I, J, order="F"),
        F,
        converged,
    )
