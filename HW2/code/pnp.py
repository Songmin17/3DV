import numpy as np

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    
    '''
    Algorithm:
    1. Use DLT to find the matrix A in the DLT setup that transforms X to x. 
    Matrix is defined in the lecture slides. 
        1a. Create a lambda function called fill_row, that takes i, xyind as input
        where i indicates the ith row, and xyind indicates whether to use 0th or 1st coord of x.
        1b. Generate all distinct pairs of ith row and xyind and store this in 'pairs'.
        1c. Apply the lambda func fill_row over all 'pairs' to get the matrix A. 
    2. The camera matrix is the vector p that minimizes Ap = 0. Find p via SVD and reshape it into matrix P.
    P = K[R | t]. But here, since we deal with normalized x (inverse(K) is already applied to x),
    in the context of normalized coordinates, P = [R | t].
    3. Scale down P, using the largest singular value (or just do U @ I @ Vt, where U, Vt are found via SVD on P).
    Then extract R, t.
    4. Change the signs of R, t if det(R) is negative.
    '''
    # TODO Your code goes here
    # Step 1: Create the DLT matrix P, where x[i] = P @ X[i]
    num_samples = X.shape[0]
    hX = np.column_stack((X, np.ones(num_samples)))
    fill_row = lambda i, xyind : (np.concatenate(((1-xyind) * hX[i] + np.zeros(4), xyind * hX[i] + np.zeros(4), -x[i, xyind] * hX[i]))).tolist() 
    list1, list2 = range(num_samples), range(2)
    pairs = np.transpose((np.tile(list1, len(list2)), np.repeat(list2, len(x))))
    pairs = list(map(tuple, pairs))

    A = list(map(lambda p: fill_row(*p), pairs))
    A = np.asarray(A)
    
    # Step 2
    _, _, Vt = np.linalg.svd(A)
    pvec = Vt[-1]
    P = np.reshape(pvec, (3, 4))

    # Step 3
    up, dp, vtp = np.linalg.svd(P[:, 0:3])
    R = up @ vtp
    d = dp[0]
    t = P[:, 3] / d
    
    # Step 4
    if np.linalg.det(R) < 0:
        R, t = -R, -t

    C = -R.T @ t

    return R, C


def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    
    '''
    Algorithm:
    1. Transform X, x to homogenous coordinates. 
    2. In each RANSAC iteration, choose at least 3 random indices. I chose 10 to improve accuracy.
    3. Use PnP on the randomly sampled X, x pairs to estimate new R, C. 
    4. Construct the camera matrix from the R, C. 
    5. Apply the camera matrix on hX to obtain 2d projections. 
        Implementation detail: as hX is an aggregation of the transposes of X,
        result = hX @ cam_matrix.T
    6. check that projections are valid via cheirality. In particular, check if the result has positive depth.
    7. Filter the 2d points based on the condition in step 6. 
    8. Compute the reprojection error and filter the points where norms are below the threshold. 
    9. If the # of points found in #9 is the current max, update final R, C.

    '''
    # TODO Your code goes here
    # Step 1
    n = X.shape[0]
    maxcount, inlier, finalR, finalC = 0, np.zeros((n,), dtype=int), np.eye(3), np.zeros(3)
    hX, hx = np.concatenate((X, np.ones((n, 1))), axis=1), np.concatenate((x, np.ones((n, 1))), axis=1)

    for _ in range(ransac_n_iter):
        curr_inlier = np.zeros((n,), dtype=int)
        # Step 2
        random_indices = np.random.choice(n, size=10, replace=False)
        sampled_X, sampled_x = X[random_indices], x[random_indices]
        # Step 3
        R, C = PnP(sampled_X, sampled_x)
        # Step 4
        cam_matrix = R @ np.concatenate((np.eye(3), -1 * np.expand_dims(C, axis=1)), axis=1)
        # Step 5
        transposed_result = hX @ cam_matrix.T 
        # Step 6: A point is in front of the camera if its reprojected 2d point's depth (3rd coord) is positive.
        valid_indices = np.where(transposed_result[:, 2] > 0)[0]
        # Step 7
        filtered_transposed_result = transposed_result[valid_indices]
        # Step 8
        filtered_transposed_result /= np.expand_dims(filtered_transposed_result[:, 2], axis=1)
        norms = np.linalg.norm((filtered_transposed_result - hx[valid_indices]), axis=1)
        inlier_indices = np.where(norms < ransac_thr)[0]
        count = len(inlier_indices)
        # Step 9
        if count > maxcount:
            curr_inlier[inlier_indices] = 1
            maxcount = count 
            inlier = curr_inlier 
            finalR, finalC = R, C

    return finalR, finalC, inlier

def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0,:]
    dv_dc = -R[1,:]
    dw_dc = -R[2,:]
    # df_dc is in shape (2, 3)
    df_dc = np.stack([
        (w * du_dc - u * dw_dc) / (w**2),
        (w * dv_dc - v * dw_dc) / (w**2)
    ], axis=0)

    du_dR = np.concatenate([X-C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X-C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack([
        (w * du_dR - u * dw_dR) / (w**2),
        (w * dv_dR - v * dw_dR) / (w**2)
    ], axis=0)


    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray([
        [0, 0, -4*qy, -4*qz],
        [-2*qz, 2*qy, 2*qx, -2*qw],
        [2*qy, 2*qz, 2*qw, 2*qx],
        [2*qz, 2*qy, 2*qx, 2*qw],
        [0, -4*qx, 0, -4*qz],
        [-2*qx, -2*qw, 2*qz, 2*qy],
        [-2*qy, 2*qz, -2*qw, 2*qx],
        [2*qx, 2*qw, 2*qz, 2*qy],
        [0, -4*qx, -4*qy, 0],
    ])

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis,:]) @ R_i.T
        proj = proj[:,:2] / proj[:,2,np.newaxis]

        H = np.zeros((7,7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j,:])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j,:] - proj[j,:])
        
        delta_p = np.linalg.inv(H + lamb*np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)


    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined