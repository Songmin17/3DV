import numpy as np

def reconstructionError(predicted_joints, gt_joints): 
  # print(f'Debug recon error:\n{predicted_joints.shape}\n{gt_joints.shape}')
  # L2 Error btw 2d keypoints, distance funct. sqrt of sum of squared error
  # pj_indices are the ith joints in 'predicted_joints' that correspond to the 
  # 0th, 1st, 2nd, ..., 14th joint in gt_joints
  pj_indices = np.array([8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 24])
  sampled_pjt = predicted_joints[pj_indices]
  norms = np.linalg.norm((sampled_pjt - gt_joints), axis=1)
  norm_sum = np.sum(norms ** 2)
  error = np.sqrt(norm_sum)
  return error

def avgReconError(recon_errors):
  return np.mean(recon_errors)