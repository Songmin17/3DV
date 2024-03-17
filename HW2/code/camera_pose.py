import numpy as np

from feature import EstimateE_RANSAC
from utils import find_indices

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    # TODO Your code goes here
    
    '''
    Algorithm:
    Based on Lecture Slides
    1. rotation matrix R = U @ D @ Vt, where U, Vt are 
    orthogonal matrices derived from E's SVD, and D = W or W.T.
    Translation vector t = third column of U.
    2. If rotation matrix's determinant < 0, change the sign of R and t so that det >= 0.
    3. cam center C = (transpose R) @ t, where t = +/-(last column of U)
    '''

    # Step 1
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    Wt = W.T 
    t1 = U[:, 2]
    t2 = -t1
    rmat1, rmat2 = U @ W @ Vt, U @ Wt @ Vt

    # Step 2
    rmat1 = -rmat1 if np.linalg.det(rmat1) < 0 else rmat1 
    rmat2 = -rmat2 if np.linalg.det(rmat2) < 0 else rmat2 
    R_set = np.asarray([rmat1, rmat1, rmat2, rmat2])
    t_set = np.asarray([t1, t2, t1, t2])

    # Step 3
    t_set = np.expand_dims(t_set, axis=-1)
    R_t_set = np.transpose(R_set, [0, 2, 1])
    C_set = np.matmul(-R_t_set, t_set)

    return R_set, C_set

def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    
    '''
    Algorithm:
    Based on lecture slides
    1. let PX = Projection of X on image plane 1. Then x (cross product) PX = 0.
    Derive the skew-symmetric matrix for this corss product -> store in 'pts'
    2. Least squares solution to find X. Use SVD to find final column of Vt.
    3. Convert X to nonhomog coord by dividing it by the final scale-up term.
    '''
    # TODO Your code goes here
    x_pts_1, x_pts_2 = track1[:, 0], track2[:, 0]
    y_pts_1, y_pts_2 = track1[:, 1], track2[:, 1]
    
    unum = track1.shape[0]
    
    # Step 1
    pts = list(map(lambda i: np.asarray([x_pts_1[i] * P1[-1] - P1[0],
                              y_pts_1[i] * P1[-1] - P1[1],
                              x_pts_2[i] * P2[-1] - P2[0],
                              y_pts_2[i] * P2[-1] - P2[1]
                              ]), range(unum)))

    # Step 2
    X = np.zeros((unum, 3))
    for i in range(len(pts)):
        _, _, Vt = np.linalg.svd(pts[i])
        # Step 3
        x = Vt[-1]
        x /= x[-1]
        x = x[:3]
        X[i, :] = x

    return X


# reference: https://cmsc426.github.io/sfm/
def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """

    '''
    Algorithm:
    1. Find R and cam center C for all cameras.
    2. X is in front of the camera if product btw position of X wrt 
    cam_center and final row of R (depth row) is positive.
    Find all such X for cam1. Repeat for cam2.
    3. Choose only the X that are in front of both cameras. 

    '''
    
    # TODO Your code goes here
    # Step 1
    R1, t1 = P1[:, :3], P1[:, 3]
    R2, t2 = P2[:, :3], P2[:, 3]
    c1, c2 = -R1.T @ t1, -R2.T @ t2 
    
    # Step 2
    temp1 = (X - c1) @ R1[-1].T 
    temp2 = (X - c2) @ R2[-1].T 

    # Step 3
    valid_index = (temp1 > 0) & (temp2 > 0)

    return valid_index


def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    
    '''
    Algorithm:
    1. Find valid track indices (!= -1) that correspond to the same world coordinate (same ith position on both tracks)
    2. Use these filtered track points to estimate E.
    3. Derive candidate R and C from E.
    4. Find the best camera pose P from the given candidates. 
    Camera pose is good if it is used to find the largest # of valid 3D points, given the valid track indices.
    '''
    # TODO Your code goes here
    # TODO: filter track1, track2 for indices where both are != [-1, -1]

    # Step 1
    ind1, ind2 = find_indices(track1), find_indices(track2)
    indices = np.where((ind1 & ind2) == True)[0]
    track1_pts, track2_pts = track1[indices], track2[indices]   # tracki_pts: parts of track that != [-1, -1]

    # Step 2
    ransac_n_iter, ransac_thr = 200, 0.01    
    E, inlier = EstimateE_RANSAC(track1_pts, track2_pts, ransac_n_iter, ransac_thr)     # inlier: the set of indices within the filtered track
  
    # Step 3
    R_set, C_set = GetCameraPoseFromE(E)
    I, t1 = np.eye(3), np.zeros((3,1))
    P1 = np.concatenate((I, t1), axis=1)
    
    # Step 4
    P2_set = []
    max_valid_count, ith_pose = -1, -1
    filteredX = None
    for i in range(4):
        # construct P from current R, C
        P2 = R_set[i] @ np.concatenate((I, -C_set[i]), axis=1)
        tempX = Triangulation(P1, P2, track1_pts[inlier], track2_pts[inlier])
        valid_index_i = EvaluateCheirality(P1, P2, tempX) # indices[inlier] -> all triangulated pts, valid_index_i = indices among all tri pts that are valid
        valid_count_i = int(np.sum(valid_index_i))

        if valid_count_i > max_valid_count:
            ith_pose = i 
            max_valid_count = valid_count_i
            filteredX = tempX
        P2_set.append(P2)

    X = np.empty((track1.shape[0], 3))
    X.fill(-1)
    X[indices[inlier]] = filteredX

    R, C = R_set[ith_pose], C_set[ith_pose]

    return R, C, X
