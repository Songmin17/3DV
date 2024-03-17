import cv2
import numpy as np
import random

from sklearn.neighbors import NearestNeighbors


threshold = 0.70
def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    # TODO Your code goes here
    '''
    Algorithm:
    1. Details are the same as in hw1. Find SIFT correspondences of points in image 1 in image 2. 
    Do this for the opposite direction. one_directional_nn for both.
    2. Choose only the pairs of correspondences present in both. This means that a pair of correspondences
    is considered a match, in the perspective of both images. 
    
    '''
    # Step 1
    def one_directional_nn(d1, d2):
        pairs = []
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors.fit(d2)
        for i in range(len(d1)):
            candidate_matches = neighbors.kneighbors([d1[i]], n_neighbors=2, return_distance=True)
            # print(f'candidates for {i}th elem:\n{candidate_matches}')
            distances = np.squeeze(candidate_matches[0])
            indices = np.squeeze(candidate_matches[1])
            dst1, dst2 = distances[0], distances[1]

            img1_index = i
            img2_index = indices[0]

            ratio = dst1 / dst2
            if ratio <= threshold:
                # print('ratio', ratio)
                pairs.append((img1_index, img2_index))
        return pairs

    d1_to_d2 = one_directional_nn(des1, des2)
    d2_to_d1 = one_directional_nn(des2, des1)
    d2_to_d1 = [(elem[1], elem[0]) for elem in d2_to_d1]

    # Step 2
    final_pairs = list(set(d1_to_d2) & set(d2_to_d1))
    final_pairs = np.array(final_pairs)

    index1, index2 = final_pairs[:, 0], final_pairs[:, 1]
    loc1, loc2 = np.asarray(loc1), np.asarray(loc2)
    img1_pt = loc1[index1]
    img2_pt = loc2[index2]
    return img1_pt, img2_pt, index1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    
    '''
    Algorithm:
    1. Find DLT matrix A in lecture slides
    2. SVD solution that minimizes Ae = 0 is the vector form of essential matrix E.
    3. Transform e to E and scale it down. E = U @ D @ Vt, where D is given in the HW2 handout. 
    
    '''
    # TODO Your code goes here
    # Step 1
    maxy, _ = x1.shape
    # assumption: maxy could be less than 8, so num_samples may have to be less than 8. 
    num_samples = min(maxy, 8)
    A = list(map(lambda i: [x2[i, 0] * x1[i, 0], 
                            x2[i, 0] * x1[i, 1],
                            x2[i, 0],
                            x2[i, 1] * x1[i, 0],
                            x2[i, 1] * x1[i, 1],
                            x2[i, 1],
                            x1[i, 0],
                            x1[i, 1],
                            1], range(num_samples)))
    A = np.array(A)
    # Step 2
    _, _, Vt = np.linalg.svd(A)
    e = Vt[-1]
    # Step 3
    E = np.reshape(e, (3, 3))
    U, _, eVt = np.linalg.svd(E)
    S = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    newE = U @ S @ eVt 

    return newE


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    '''
    Algorithm:
    1. Randomly sample 8 (or fewer, if # of given points is smaller than 8) points from X, x.
    2. Estimate E using these X, x pairs. 
    3. Check if E is good enough. Minimize (x.T) @ E @ x = 0 for all x. 
    4. Find x that pass this test. 
    5. Find E estimate that maximizes # of such x. 
    
    '''
    # TODO Your code goes here
    # Step 1
    maxy, _ = x1.shape
    ones = np.ones(maxy)
    hx1 = np.column_stack((x1, ones))
    hx2 = np.column_stack((x2, ones))

    maxcount = -1
    E, inlier = None, None 
    for _ in range(ransac_n_iter):
        # Step 2
        # assumption: maxy could be less than 8, so num_samples may have to be less than 8. 
        num_samples = min(maxy, 8)
        random_indices = np.random.choice(x1.shape[0], size=num_samples, replace=False)
        sampled_x1, sampled_x2 = x1[random_indices], x2[random_indices]
        # Step 3
        tempE = EstimateE(sampled_x1, sampled_x2)
        res1 = hx2 @ tempE @ hx1.T
        values = np.diagonal(res1)
        norms = np.sqrt(values * values)
        # Step 4
        indices = np.where(norms < ransac_thr)
        counts = len(indices[0])
        # Step 5
        if counts > maxcount:
            E = tempE
            inlier = indices[0]
            maxcount = counts

    return E, inlier


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """

    '''
    Algorithm:
    1. Extract all features.
    2. For every ith image, initialize track_i.
    3. Perform RANSAC on every distinct pair of images. 
    4. Filter matched points based on threshold. For the kth keypoints in ith image 
    that are matched, store their coordinates, and in the jth row, store the coordinates of the 
    correspondence point in the jth image. 
    5. Filter track_i to keep just the valid coordinates (coord != -1).
    6. Append track_i to track. 
    '''
    
    # TODO Your code goes here
    # Step 1
    N, H, W, chan = Im.shape
    grayIm = np.array([cv2.cvtColor(Im[i], cv2.COLOR_RGB2GRAY) for i in range(N)])
    sift = cv2.SIFT_create()
    keypts = np.array([sift.detect(grayIm[i]) for i in range(N)], dtype=object)
    loc_des = np.array([sift.compute(grayIm[i], keypts[i]) for i in range(N)], dtype=object)

    loc = np.array([[loc_des[i][0][j].pt for j in range(len(loc_des[i][0]))] for i in range(len(loc_des))], dtype=object)
    des = loc_des[:, 1]

    # check if SIFT descriptors were properly extracted for all images

    f = np.array([len(keypts[i]) for i in range(N)])
    track = None

    ransac_n_iter, ransac_thr = 200, 0.01
    for i in range(N-1):
        # Step 2
        fcurr = f[i]
        print(f'fcurr: {fcurr}')
        track_i = np.zeros((N, fcurr, 2))
        track_i.fill(-1)

        matched_indices = None
        for j in range(i+1, N):
            # Step 3
            x1, x2, indices = MatchSIFT(loc[i], des[i], loc[j], des[j])
            maxy = x1.shape[0]
            ones = np.ones(maxy)
            hx1, hx2 = np.column_stack((x1, ones)), np.column_stack((x2, ones))
            invK = np.linalg.inv(K)
            normalized_x1, normalized_x2 = hx1 @ invK.T, hx2 @ invK.T
            normalized_x1, normalized_x2 = normalized_x1[:, 0:2], normalized_x2[:, 0:2]            
            E, inlier_indices = EstimateE_RANSAC(normalized_x1, normalized_x2, ransac_n_iter, ransac_thr)

            # update track_i using inlier matches
            
            # Step 4
            curr_indices = indices[inlier_indices]
            track_i[i][curr_indices] = normalized_x1[inlier_indices]
            track_i[j][curr_indices] = normalized_x2[inlier_indices]

            if matched_indices is None:
                matched_indices = curr_indices 
            else:
                matched_indices = np.concatenate((matched_indices, curr_indices))
            
        # Step 5
        matched_indices = list(set(matched_indices))
        track_i = track_i[:, matched_indices, :]
        if track is None:
            track = track_i
        else:
            track = np.concatenate((track, track_i), axis=1)
    
    return track


