import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment

from utils import find_indices


if __name__ == '__main__':
    np.random.seed(100)
    K = np.asarray([
        [463.1, 0, 333.2],
        [0, 463.1, 187.5],
        [0, 0, 1]
    ])
    num_images = 14
    w_im = 672
    h_im = 378

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        # im_file = 'images/image{:01d}.jpg'.format(i + 1)
        im_file = 'images/image{:01d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    # Build feature track
    # try:
    #     with open('track.npy', 'rb') as f:
    #         track = np.load(f)
    # except:
    #     track = BuildFeatureTrack(Im, K)

    # print(f'track size: {track.shape}')
    #############################################################################

    # TODO: get rid of debug routine
    # print(f'final track shape: {track.shape}')

    # vis = np.concatenate(Im, axis=1)
    # maxy, maxx, _ = vis.shape

    # track_i_indices = random.sample(range(0, track.shape[1]), 5)

    # lines = []
    # for ind in track_i_indices:
    #     sublines = []
    #     curr_pt, curr_ind = track[0, ind, :], 1
    #     print(f'curr_pt shape: {curr_pt.shape}\ncurr_pt: {curr_pt}')
    #     curr_pt = np.append(curr_pt, 1)
    #     curr_pt = (K @ curr_pt)[0:2]
        
    #     while curr_ind < 6:
    #         next_pt = track[curr_ind, ind, :]
    #         next_pt = np.append(next_pt, 1)
    #         # assume that next_pt = (x, y) form
    #         if next_pt[0] != -1:
    #             next_pt = (K @ next_pt)[0:2]
    #             next_pt[0] = w_im * curr_ind + next_pt[0]
    #             # print(f'curr_pt, next_pt:\n{int(curr_pt / 2)}\n{int(next_pt / 2)}')
    #             sublines.append([(curr_pt / 2).astype(int), (next_pt / 2).astype(int)])
    #             curr_pt = next_pt
    #         curr_ind += 1
        
    #     lines.append(sublines)

    # print(f'lines:\n{lines}')

    # vis = cv2.resize(vis, (maxx//2, maxy//2))
    # img_with_line = vis.copy()
    # colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
    # pink = (255, 192, 203)

    # for i, subline in enumerate(lines):
    #     for subsubline in subline:
    #         img_with_line = cv2.circle(img_with_line, subsubline[0], 10, colors[i % 5], -1)
    #         img_with_line = cv2.circle(img_with_line, subsubline[1], 10, colors[i % 5], -1)
    #         img_with_line = cv2.line(img_with_line, subsubline[0], subsubline[1], colors[i % 5], 2)

    # cv2.imshow('side by side', img_with_line)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #############################################################################

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))

    # Set first two camera poses
    # TODO Your code goes here
    I = np.eye(3)
    P[0, :, :3] = I
    P[1, :, :] = R @ np.concatenate((I, -C), axis=1)

    ransac_n_iter = 200
    ransac_thr = 0.01 
    for i in range(2, num_images):
        # Estimate new camera pose
        track_i = track[i, :, :]
        # find the indices of X, track_i where X and correspondence point x are both found. 
        valid_indices = np.logical_and(X[:,0] != -1, track_i[:,0] != -1)
        filtered_X, filtered_track_i = X[valid_indices, :], track_i[valid_indices, :]

        # Add new camera pose to the set and make a few edits to it. 
        R_i, C_i, inlier_indices = PnP_RANSAC(filtered_X, filtered_track_i, ransac_n_iter, ransac_thr)
        print(f'# of inlier indices, total indices: {np.sum(inlier_indices)}, {inlier_indices.size}')
        updated_R_i, updated_C_i = PnP_nl(R_i, C_i, filtered_X[inlier_indices], filtered_track_i[inlier_indices])
        P[i] = updated_R_i @ np.concatenate((I, -1 * np.expand_dims(updated_C_i, axis=1)), axis=1)

        for j in range(i):
            # Find new points to reconstruct
            track_j = track[j, :, :] 
            # find ith X points that aren't yet reconstructed but have correspondence points in both images. 
            missing_indices = FindMissingReconstruction(X, track_i).astype(int) * FindMissingReconstruction(X, track_j).astype(int)
            missing_indices = missing_indices.astype(bool)
            filtered_tracki, filtered_trackj = track_i[missing_indices], track_j[missing_indices]
            # create 3D points from just the filtered 2d points. 
            missingX = Triangulation(P[i], P[j], filtered_tracki, filtered_trackj)
            missingX = Triangulation_nl(missingX, P[i], P[j], filtered_tracki, filtered_trackj)
            # check if the new 3D points are in front of both cameras. 
            valid_indices_mx = EvaluateCheirality(P[i], P[j], missingX)

            # Update 3D points
            missing_indices = np.flatnonzero(missing_indices)
            X[missing_indices[valid_indices_mx], :] = missingX[valid_indices_mx]

            print('{} -> {} camera: {} new points are found'.format(i+1, j+1, valid_indices_mx.size))
            
        # Run bundle adjustment
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i + 1, :, :] = P_new
        X[valid_ind, :] = X_new

        P[:i+1,:,:] = P_new
        X[valid_ind,:] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)