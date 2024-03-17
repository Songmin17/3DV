import numpy as np
import scipy.linalg

np.random.seed(0)

# svd
def svd2():
    print('*' * 50)
    A = np.array([[2, 3, 3], [5, 3, 1], [4, 5, 1]])
    U, S, VT = np.linalg.svd(A)
    Sigma = np.diag(S)
    print('SVD of the given matrix A')
    print(f'A: \n{A}\nU:\n{U}\nSigma:\n{Sigma}\nVT:\n{VT}')
    print(f'U*Sigma*VT = \n{U@Sigma@VT}')
    print('A is the given matrix. U and VT are orthogonal matrices, and Sigma is a diagonal matrix of singular values for A.')
    # TODO: show that U, V are orthogonal
    print(f'U is orthogonal because UT * U = U * UT = I')
    print(f'UT * U =\n{np.rint(U.T @ U)}')
    print(f'U * UT =\n{np.rint(U @ U.T)}')

    print(f'V is orthogonal because VT * V = V * VT = I')
    print(f'VT * V =\n{np.rint(VT @ VT.T)}')
    print(f'V * VT =\n{np.rint(VT.T @ VT)}')

    print('Extremely small decimal values in the matrices were rounded to zero.')
    print('This is why some of the zeros in the matrices retain the negative sign.')
    print('*' * 50)

    m1_size = (5, 3)
    M1 = np.random.randint(5, size=m1_size)
    MU1, S, MVT1 = np.linalg.svd(M1)
    MS1 = np.zeros(m1_size)
    MS1[:3, :3] = np.diag(S)
    print('SVD of random (5, 3) matrix')
    print(f'matrix A: \n{M1}\nU:\n{MU1}\nSigma:\n{MS1}\nVT:\n{MVT1}')
    print(f'U*Sigma*VT = \n{np.rint(MU1@MS1@MVT1)}')
    print(f'This matrix is equal to A.')
    # TODO: show that U, V are orthogonal
    print(f'U is orthogonal because UT * U = U * UT = I')
    print(f'UT * U =\n{np.rint(MU1.T @ MU1)}')
    print(f'U * UT =\n{np.rint(MU1 @ MU1.T)}')

    print(f'V is orthogonal because VT * V = V * VT = I')
    print(f'VT * V =\n{np.rint(MVT1 @ MVT1.T)}')
    print(f'V * VT =\n{np.rint(MVT1.T @ MVT1)}')

    print('*' * 50)
    m2_size = (3, 4)
    M2 = np.random.randint(10, size=m2_size)
    MU2, S, MVT2 = np.linalg.svd(M2)
    MS2 = np.zeros(m2_size)
    MS2[:3, :3] = np.diag(S)
    print('SVD of random (3, 4) matrix')
    print(f'matrix A: \n{M2}\nU:\n{MU2}\nSigma:\n{MS2}\nVT:\n{MVT2}')
    print(f'U*Sigma*VT = \n{np.rint(MU2@MS2@MVT2)}')
    print(f'This matrix is equal to A.')
    # TODO: show that U, V are orthogonal
    print(f'U is orthogonal because UT * U = U * UT = I')
    print(f'UT * U =\n{np.rint(MU2.T @ MU2)}')
    print(f'U * UT =\n{np.rint(MU2 @ MU2.T)}')

    print(f'V is orthogonal because VT * V = V * VT = I')
    print(f'VT * V =\n{np.rint(MVT2 @ MVT2.T)}')
    print(f'V * VT =\n{np.rint(MVT2.T @ MVT2)}')

def qr3():
    print('*' * 50)
    A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    q, r = np.linalg.qr(A)
    print('QR Decomposition of the given matrix A')
    print(f'A: \n{A}\nQ:\n{q}\nR:\n{r}')
    print('Q is an orthogonal matrix because QT * Q = Q * QT = I')
    print(f'QT * Q =\n{np.rint(q.T @ q)}')
    print(f'Q * QT =\n{np.rint(q @ q.T)}')

def cholesky4():
    print('*' * 50)
    A = np.array([[1, -1, 2], [-1, 5, -4], [2, -4, 6]])
    l = np.linalg.cholesky(A)
    print('Cholesky Decomposition of the given matrix A')
    print(f'A: \n{A}\nL:\n{l}\nConjugate Transpose of L:\n{l.T.conj()}')
    print('Verification that L * (Conjugate Transpose of L) = A')
    print(f'{l} times\n{l.T.conj()} =\n{np.rint(l @ l.T.conj())} =\n{A}')

def lu5():
    print('*' * 50)
    A = np.array([[2, 1, 3], [4, -1, 3], [-2, 5, 5]])
    p, l, u = scipy.linalg.lu(A)
    print('LU Decomposition of the given matrix A')
    print(f'A:\n{A}\nP:\n{p}\nL:\n{l}\nU:\n{u}')
    print('A is the given matrix. P is the permutation matrix. L and U are the component matrices of A.')

def homog_eq6():
    print('*' * 50)
    A = np.array([[-3.0975, -1.9844, 3.9311], 
                  [14.0057, 7.9991, -17.9960], 
                  [3.9054, 2.0152, -5.0669], 
                  [1.9940, 78.0010, -1.0043]])
    u, s, vt = np.linalg.svd(A)
    x = vt[-1]
    print('Solution to the homogenous linear system Ax = 0 using SVD')
    print(f'Solution:\n{x}')
    print(f'Verification:\nA * x = {A @ x}')
    print(f'When rounded to zero, the value of A * x becomes\n{np.rint(A @ x)}')


def linearlstsq7():
    print('*' * 50)
    A = np.array([[1, 1, 1],
                  [6, 0, 3],
                  [2, 1, 2],
                  [1, 8, 0]])
    b = np.array([[8.2593],
                  [17.9659],
                  [15.9216],
                  [3.1024]])
    
    u, s, vt = np.linalg.svd(A)
    invSigma = np.zeros((A.shape[1], A.shape[0]))
    
    # print(f'invSigma:\n{invSigma}')
    inv_s = np.asarray([1/s[i] if s[i] != 0 else 0 for i in range(len(s))])

    dim = min(A.shape[0], A.shape[1])
    invSigma[0:dim, 0:dim] = np.diag(inv_s)
    pseudo_inv = vt.T @ invSigma @ u.T
    # Ax = b 
    # x = pseu_inv * b
    print(f'#7: Solution to Ax = b using SVD')
    print(f'A = \n{A}\nb = \n{b}')
    print(f'pseudo-inverse T:\n{pseudo_inv}')
    x = pseudo_inv @ b
    print(f'T * b =\n{x}')
    print(f'= X, the solution')
    print('Verifying...')
    print(f'A * X =\n{A @ x} = b = \n{b}')


if __name__ == '__main__':
    # svd2()
    # qr3()
    # cholesky4()
    # lu5()
    # homog_eq6()
    linearlstsq7()