import cv2
import math 
import numpy as np
import random

from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RectBivariateSpline 

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
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    x1 = []
    x2 = []
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(des2)

    for i in range(len(des1)):
        candidate_matches = neighbors.kneighbors([des1[i]], n_neighbors=2, return_distance=True)
        # print(f'candidates for {i}th elem:\n{candidate_matches}')
        distances = np.squeeze(candidate_matches[0])
        indices = np.squeeze(candidate_matches[1])
        d1, d2 = distances[0], distances[1]

        img1_index = i
        img2_index = indices[0]
        img1_pt = loc1[img1_index]
        img2_pt = loc2[img2_index]
        ratio = d1 / d2
        if ratio <= threshold:
            x1.append(img1_pt)
            x2.append(img2_pt)
        
    return x1, x2 

def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # TODO Your code goes here
    # randomly sample 4 indices i, j, l, m
    random.seed(2)

    h_bank = []
    inlier_indices = []
    inlier_count = [] 
    max_inlier_count, max_index = -1, -1

    for myIter in range(ransac_n_iter):
        sampled_indices = random.sample(list(range(len(x1))), 4)
        i, j, l, m = sampled_indices[0], sampled_indices[1], sampled_indices[2], sampled_indices[3]
        # create A 
        A = np.array([[x1[i][0], x1[i][1], 1, 0, 0, 0, -x1[i][0]*x2[i][0], -x1[i][1]*x2[i][0], -x2[i][0]],
                    [0, 0, 0, x1[i][0], x1[i][1], 1, -x1[i][0]*x2[i][1], -x1[i][1]*x2[i][1], -x2[i][1]],
                    [x1[j][0], x1[j][1], 1, 0, 0, 0, -x1[j][0]*x2[j][0], -x1[j][1]*x2[j][0], -x2[j][0]],
                    [0, 0, 0, x1[j][0], x1[j][1], 1, -x1[j][0]*x2[j][1], -x1[j][1]*x2[j][1], -x2[j][1]],
                    [x1[l][0], x1[l][1], 1, 0, 0, 0, -x1[l][0]*x2[l][0], -x1[l][1]*x2[l][0], -x2[l][0]],
                    [0, 0, 0, x1[l][0], x1[l][1], 1, -x1[l][0]*x2[l][1], -x1[l][1]*x2[l][1], -x2[l][1]],
                    [x1[m][0], x1[m][1], 1, 0, 0, 0, -x1[m][0]*x2[m][0], -x1[m][1]*x2[m][0], -x2[m][0]],
                    [0, 0, 0, x1[m][0], x1[m][1], 1, -x1[m][0]*x2[m][1], -x1[m][1]*x2[m][1], -x2[m][1]] 
                    ])
        
        _, _, VT = np.linalg.svd(A)
        h = VT[-1]
        H = np.reshape(h, (3, 3))
        H = np.divide(H, H[2][2])
        if np.linalg.det(H) < 0:
            H = -1 * H

        p1 = (np.c_[x1, np.ones(len(x1))]).T
        p2 = (np.c_[x2, np.ones(len(x2))]).T
        
        checkH1 = H @ p1
        y, x = checkH1.shape 
        for pt in range(x):
            checkH1[:, pt] /= checkH1[2, pt]

        column_error = np.linalg.norm((checkH1-p2), axis=0) 
        column_error[column_error <= ransac_thr] = 1 
        column_error[column_error > ransac_thr] = 0 

        num_inliers = np.sum(column_error)
        inlier_count.append(num_inliers)
        h_bank.append(H)
        my_inliers = np.where(column_error == 1)
        inlier_indices.append(my_inliers)

        if num_inliers > max_inlier_count: 
            max_inlier_count = num_inliers 
            max_index = myIter 

    return h_bank[max_index], np.squeeze(inlier_indices[max_index])

def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    
    # TODO Your code goes here
    E = np.linalg.inv(K) @ H @ K
    return E 

def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Wc, 3)
        The 3D points corresponding to all pixels in the canvas
    """
    # TODO Your code goes here
    f = K[0][0]
    x = np.linspace(0, Wc-1, Wc)
    phi = (x * 2 * math.pi) / Wc
    new_x = f * np.cos(phi)
    y = np.linspace(-Hc//2, Hc//2, Hc)
    z = f * np.sin(phi)

    mx, my = np.meshgrid(new_x, y)
    mz, my = np.meshgrid(z, y)
    result = np.array([[mz], [my], [mx]]) 
    result = np.squeeze(result)
    return result

def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    
    # TODO Your code goes here
    maxx, maxy, maxz = p.shape
    mask = np.zeros((maxy, maxz), dtype=int)
    u = np.zeros((maxy, maxz, 2))
    R_t = np.zeros((3, 4))
    R_t[:, 0:3] = R 

    for y in range(maxy):
        for z in range(maxz):
            point = p[:, y, z]
            point_homog = np.array([point[0], point[1], point[2], 1])
            rotated_point_homog = R_t @ point_homog
            projected_pt = K @ rotated_point_homog
            projected_pt /= projected_pt[-1]
            u[y, z, 0] = projected_pt[0]
            u[y, z, 1] = projected_pt[1]

            if rotated_point_homog[2] <= 0:
                continue 
            if projected_pt[0] >= 0 and projected_pt[0] < W and projected_pt[1] >= 0 and projected_pt[1] < H: 
                mask[y, z] = 1

    return u, mask

def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    
    # TODO Your code goes here
    h, w, z = image_i.shape
    print(f'h, w, z: {h}, {w}, {z}')
    h_arr = np.linspace(0, h-1, h)
    w_arr = np.linspace(0, w-1, w)
    b_img, g_img, r_img = image_i[:, :, 0],  image_i[:, :, 1],  image_i[:, :, 2]

    # TODO: get rid of this after check
    u = u.astype(int)
    image_i_spline_b = RectBivariateSpline(h_arr, w_arr, b_img)
    image_i_spline_g = RectBivariateSpline(h_arr, w_arr, g_img)
    image_i_spline_r = RectBivariateSpline(h_arr, w_arr, r_img)
    h_coord, w_coord = u[:, :, 1], u[:, :, 0]
    interp_spline_b = image_i_spline_b.ev(h_coord, w_coord)
    interp_spline_g = image_i_spline_g.ev(h_coord, w_coord)
    interp_spline_r = image_i_spline_r.ev(h_coord, w_coord)
    
    uh, uw, _ = u.shape
    canvas_i = np.zeros((uh, uw, 3), dtype=np.float64)
    canvas_i[:, :, 0] = interp_spline_b * mask_i 
    canvas_i[:, :, 1] = interp_spline_g * mask_i 
    canvas_i[:, :, 2] = interp_spline_r * mask_i 
    
    return (np.rint(canvas_i)).astype(np.uint8)

def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    
    # TODO Your code goes here

    inverse_mask = np.zeros_like(mask_i)
    inverse_mask.fill(1)
    inverse_mask = inverse_mask - mask_i
    canvas = canvas * inverse_mask[:, :, np.newaxis] + canvas_i * mask_i[:, :, np.newaxis]

    return canvas

if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
		# TODO Your code goes here
        gray1 = cv2.cvtColor(im_list[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(im_list[i+1], cv2.COLOR_RGB2GRAY)
		
        # Extract SIFT features
		# TODO Your code goes here
        sift = cv2.SIFT_create()
        kp1 = sift.detect(gray1)
        loc1_des1 = sift.compute(gray1, kp1)
        kp2 = sift.detect(gray2)
        loc2_des2 = sift.compute(gray2, kp2)

        loc1 = np.array([loc1_des1[0][i].pt for i in range(len(loc1_des1[0]))])
        des1 = loc1_des1[1]
        loc2 = np.array([loc2_des2[0][i].pt for i in range(len(loc2_des2[0]))])
        des2 = loc2_des2[1]
        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # TODO: get rid of this after debugging check
        img1 = im_list[i].copy()
        img2 = im_list[i+1].copy()

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)		
        R_new = R @ rot_list[i]
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)
    print(f'p shape:\n{p.shape}')

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)

    #     # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
    #     # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)