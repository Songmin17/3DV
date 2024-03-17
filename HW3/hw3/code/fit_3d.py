"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

About this Script:
============
This is a demo version of the algorithm implemented in the paper,
which fits the SMPL body model to the image given the joint detections.
The code is organized to be run on the LSP dataset.
See README to see how to download images and the detected joints.
"""

from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import pickle
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch
import eval
import hw3 

from camera import ProjectPoints
from lib.robustifiers import GMOf
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
# from render_model import render_model

_LOGGER = logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added


# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d, init_pose):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=5000.,
                      pix_thsh=25.,
                      viz=False):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image 
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be considered,
              3D vector for the body orientation
    """
    # optimize camera translation and body orientation based on torso joints
    # LSP torso ids:
    # 2=right hip, 3=left hip, 8=right shoulder, 9=left shoulder
    torso_cids = [2, 3, 8, 9]
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    # initialize camera rotation
    rt = ch.zeros(3)
    # initialize camera translation
    _LOGGER.info('initializing translation via similar triangles')
    init_t = guess_init(model, flength, j2d, init_pose)
    t = ch.array(init_t)

    # check how close the shoulder joints are
    try_both_orient = np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # initialize the camera
    cam = ProjectPoints(
        f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)

    # we are going to project the SMPL joints
    cam.v = Jtr
    on_step = None
    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        options={'maxiter': 100,
                 'e_3': .0001,
                 # disp set to 1 enables verbose output from the optimizer
                 'disp': 0})
    return (cam, try_both_orient, opt_pose[:3].r)


# --------------------Core optimization --------------------
np.random.seed(1)
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       try_both_orient,
                       body_orient,
                       n_betas=10,
                       regs=None,
                       conf=None,
                       viz=False,
                       use_pose=True,
                       use_shape=True):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image 
    :param prior: mixture of gaussians pose prior
    :param try_both_orient: boolean, if True both body_orient and its flip are considered for the fit
    :param body_orient: 3D vector, initialization for the body orientation
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 14D vector storing the confidence values from the CNN
    :param viz: boolean, if True enables visualization during optimization
    :param use_pose: boolean, if True enables pose prior
    :param use_shape: boolean, if True enables shape prior
    :returns: a tuple containing the optimized model, its joints projected on image space, the camera translation
    """
    t0 = time()
    # define the mapping LSP joints -> SMPL joints
    # cids are joints ids for LSP:
    cids = list(range(12)) + [13]
    # joint ids for SMPL
    # SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    # the vertex id for the joint corresponding to the head
    head_id = 411

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and LSP is significantly different so set
    # their weights to zero
    base_weights = np.array(
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    if try_both_orient:
        flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
            cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
        flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
        orientations = [body_orient, flipped_orient]
    else:
        orientations = [body_orient]

    if try_both_orient:
        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        errors = []

    svs = []
    cams = []
    for o_id, orient in enumerate(orientations):
        # initialize the shape to the mean shape in the SMPL training set
        betas = ch.zeros(n_betas)

        # initialize the pose by using the optimized body orientation and the
        # pose prior
        init_pose = np.hstack((orient, prior.weights.dot(prior.means)))

        # instantiate the model:
        # verts_decorated allows us to define how many
        # shape coefficients (directions) we want to consider (here, n_betas)
        sv = verts_decorated(
            trans=ch.zeros(3),
            pose=ch.array(init_pose),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs)

        # make the SMPL joints depend on betas
        Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                           for i in range(len(betas))])
        J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
            model.v_template.r)

        # get joint positions as a function of model pose, betas and trans
        (_, A_global) = global_rigid_transformation(
            sv.pose, J_onbetas, model.kintree_table, xp=ch)
        Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

        # add the head joint, corresponding to a vertex...
        Jtr = ch.vstack((Jtr, sv[head_id]))

        # ... and add the joint id to the list
        if o_id == 0:
            smpl_ids.append(len(Jtr) - 1)

        # update the weights using confidence values
        weights = base_weights * conf[
            cids] if conf is not None else base_weights

        # project SMPL joints on the image plane using the estimated camera
        cam.v = Jtr

        # data term: distance between observed and estimated joints in 2D
        obj_j2d = lambda w, sigma: (
            w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))
        # NOTE: using this term as error for reprojection? 


        # mixture of gaussians pose prior
        pprior = lambda w: w * prior(sv.pose)
        # joint angles pose prior, defined over a subset of pose parameters:
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        alpha = 10
        my_exp = lambda x: alpha * ch.exp(x)
        obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
                                                 58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])
        """Create visualization."""
            # show optimized joints in 2D
        tmp_img = img.copy()
        for coord, target_coord in zip(
                np.around(cam.r[smpl_ids]).astype(int),
                np.around(j2d[cids]).astype(int)):
            if (coord[0] < tmp_img.shape[1] and coord[0] >= 0 and
                    coord[1] < tmp_img.shape[0] and coord[1] >= 0):
                cv2.circle(tmp_img, tuple(coord), 3, [0, 0, 255])
            if (target_coord[0] < tmp_img.shape[1] and
                    target_coord[0] >= 0 and
                    target_coord[1] < tmp_img.shape[0] and
                    target_coord[1] >= 0):
                cv2.circle(tmp_img, tuple(target_coord), 3,
                            [0, 255, 0])

            cv2.imwrite('opt_viz/opt.jpg', tmp_img)
        # else:
        on_step = None

        if regs is not None:
            # interpenetration term
            sp = SphereCollisions(
                pose=sv.pose, betas=sv.betas, model=model, regs=regs)
            sp.no_hands = True
            
        # weight configuration used in the paper, with joints + confidence values from the CNN
        # (all the weights used in the code were obtained via grid search, see the paper for more details)
        # the first list contains the weights for the pose priors,
        # the second list contains the weights for the shape prior
        pose_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78] if use_pose else [0.39676747423066994, 0.538816734003357, 0.4191945144032948, 0.6852195003967595]
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1] if use_shape else [0.39676747423066994, 0.538816734003357, 0.4191945144032948, 0.6852195003967595]
        # opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
        #                   [1e2, 5 * 1e1, 1e1, .5 * 1e1])
        print(f'debug pose_weights:\n{pose_weights}\ndebug shape_weights:\n{shape_weights}')
        opt_weights = zip(pose_weights, shape_weights)

        # run the optimization in 4 stages, progressively decreasing the
        # weights for the priors
        for stage, (w, wbetas) in enumerate(opt_weights):
            _LOGGER.info('stage %01d', stage)
            objs = {}

            objs['j2d'] = obj_j2d(1., 100)

            objs['pose'] = pprior(w)

            objs['pose_exp'] = obj_angle(0.317 * w)

            objs['betas'] = wbetas * betas

            if regs is not None:
                objs['sph_coll'] = 1e3 * sp

            ch.minimize(
                objs,
                x0=[sv.betas, sv.pose],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100,
                         'e_3': .0001,
                         'disp': 0})

        t1 = time()
        _LOGGER.info('elapsed %.05f', (t1 - t0))
        if try_both_orient:
            errors.append((objs['j2d'].r**2).sum())
        svs.append(sv)
        cams.append(cam)

    if try_both_orient and errors[0] > errors[1]:
        choose_id = 1
    else:
        choose_id = 0

    return (svs[choose_id], cams[choose_id].r, cams[choose_id].t.r)


def run_single_fit(img,
                   j2d,
                   conf,
                   model,
                   regs=None,
                   n_betas=10,
                   flength=5000.,
                   pix_thsh=25.,
                   scale_factor=1,
                   viz=False,
                   do_degrees=None,
                   use_pose=True,
                   use_shape=True):
    """Run the fit for one specific image.
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    :param use_pose: boolean, if True enables pose prior
    :param use_shape: boolean, if True enables shape prior
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """
    if do_degrees is None:
        do_degrees = []

    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))

    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor

    # estimate the camera parameters
    (cam, try_both_orient, body_orient) = initialize_camera(
        model,
        j2d,
        img,
        init_pose,
        flength=flength,
        pix_thsh=pix_thsh,
        viz=viz)

    # fit
    (sv, opt_j2d, t) = optimize_on_joints(
        j2d,
        model,
        cam,
        img,
        prior,
        try_both_orient,
        body_orient,
        n_betas=n_betas,
        conf=conf,
        viz=viz,
        regs=regs,
        use_pose=use_pose,
        use_shape=use_shape)

    h = img.shape[0]
    w = img.shape[1]
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])

    images = []
    orig_v = sv.r
    for deg in do_degrees:
        if deg != 0:
            aroundy = cv2.Rodrigues(np.array([0, np.radians(deg), 0]))[0]
            center = orig_v.mean(axis=0)
            new_v = np.dot((orig_v - center), aroundy)
            verts = new_v + center
        else:
            verts = orig_v

    # return fit parameters
    params = {'cam_t': cam.t.r,
              'f': cam.f.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r,
              'opt_j2d': opt_j2d,
              'verts': verts,
              'faces': sv.f,
              'cam_rot': cam.rt.r,
              'cam_c': cam.c.r,
            }

    return params, images

index = 0
def main(base_dir,
         out_dir,
         use_interpenetration=True,
         n_betas=10,
         flength=5000.,
         pix_thsh=25.,
         use_neutral=False,
         viz=True,
         use_pose=True,
         use_shape=True):
    """Set up paths to image and joint data, saves results.
    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param use_interpenetration: boolean, if True enables the interpenetration term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (an estimate)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization
    :param use_pose: boolean, if True enables pose prior
    :param use_shape: boolean, if True enables shape prior
    """
    img_dir = join(abspath(base_dir), 'images', 'lsp')
    data_dir = join(abspath(base_dir), 'results', 'lsp')
    result_dir = join(abspath(base_dir), 'results')

    if not exists(out_dir):
        makedirs(out_dir)

    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    sph_regs = None
    if not use_neutral:
        _LOGGER.info("Reading genders...")
        # File storing information about gender in LSP
        with open(join(data_dir, 'lsp_gender.csv')) as f:
            genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs_male = np.load(SPH_REGS_MALE_PATH)
            sph_regs_female = np.load(SPH_REGS_FEMALE_PATH)
    else:
        gender = 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)

    # Load groundtruth joints
    est = np.load(join(data_dir, 'est_joints.npz'))['est_joints']

    errors = []
    # Load images
    global index 
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    for ind, img_path in enumerate(img_paths):
        # out_path = '%s/%04d.pkl' % (out_dir, ind)
        out_path = join(abspath(base_dir), 'pkl_output', f'{index:04d}.pkl')
        # print(f'pkl out_path:\n{out_path}')

        img = cv2.imread(img_path)
        if img.ndim == 2:
            _LOGGER.warn("The image is grayscale!")
            img = np.dstack((img, img, img))
        joints = est[:2, :, ind].T

        if not exists(out_path):
            _LOGGER.info('Fitting 3D body on `%s` (saving to `%s`).', img_path,
                         out_path)

            conf = est[2, :, ind]

            if not use_neutral:
                gender = 'male' if int(genders[ind]) == 0 else 'female'
                if gender == 'female':
                    model = model_female
                    if use_interpenetration:
                        sph_regs = sph_regs_female
                elif gender == 'male':
                    model = model_male
                    if use_interpenetration:
                        sph_regs = sph_regs_male

            params, vis = run_single_fit(
                img,
                joints,
                conf,
                model,
                regs=sph_regs,
                n_betas=n_betas,
                flength=flength,
                pix_thsh=pix_thsh,
                scale_factor=2,
                viz=viz,
                do_degrees=do_degrees,
                use_pose=use_pose,
                use_shape=use_shape)
            
            # save params, vis to pkl_output
            with open(out_path, 'wb') as f:
                f.write(pickle.dumps(params))

        else:   # load params, vis from pkl_output
            with open(out_path, 'rb') as f:
                params = pickle.load(f)
            joints *= 2
        
        # Visualize Optimized SMPL parameters on the Input Image
        copy = img.copy()
        color = {'green' : (113, 179, 60), 'red' : (0, 0, 255), 'blue' : (255, 0, 0), 'yellow' : (0, 234, 255)}
        gt_jt, optimized_jt = joints, params['opt_j2d']
        gt_jt = gt_jt // 2 
        optimized_jt = optimized_jt // 2 
        # print(f'gt_jt dims: {len(gt_jt)}')
        # print(f'optimized_jt dims: {len(optimized_jt)}')
        bbox_opt_jt = hw3.find_bbox(optimized_jt)
        coord0, coord2 = (bbox_opt_jt[0][0], bbox_opt_jt[0][1]), (bbox_opt_jt[2][0], bbox_opt_jt[2][1])
        img_with_opt_jt = cv2.rectangle(copy, coord0, coord2, color['red'], 2)
        img_with_opt_jt = hw3.drawKeypoints(img_with_opt_jt, optimized_jt, color['red'])
        img_with_gt_jt = hw3.drawKeypoints(img_with_opt_jt, gt_jt, color['green'])

        # TODO: draw bbox and keypoints
        cv2.imwrite(f'{result_dir}/bbox_keypoints_{index}.jpg', img_with_gt_jt)

        # NOTE: debug the coordinates on gt_jt. Color each keypoint + make image 
        # for ind, kp in enumerate(gt_jt): 
        #     im = img_with_gt_jt.copy()
        #     keypt = tuple(map(int, kp))
        #     cv2.circle(im, keypt, 1, color['blue'], -1)
        #     cv2.imwrite(f'{result_dir}/gt_{index}_{ind}.jpg', im)
        
        # TODO: draw skeleton
        img_with_skeleton = hw3.drawSkeleton(img, optimized_jt, color['red'], color['green'])
        cv2.imwrite(f'{result_dir}/skeleton_{index}.jpg', img_with_skeleton)

        # TODO: exportMesh and drawMesh
        copy2 = img.copy()
        verts, faces = params['verts'], params['faces']
        focal, cam_c, cam_rot, cam_t = params['f'], params['cam_c'], params['cam_rot'], params['cam_t']
        savepath = f'{result_dir}/mesh_{index}.obj'
        hw3.exportMesh(savepath, verts, faces)

        projected_verts = hw3.projectPoints(verts, focal, cam_c, cam_rot, cam_t)
        img_with_mesh = hw3.drawMesh(copy2, projected_verts, faces, color['yellow'])
        cv2.imwrite(f'{result_dir}/mesh_{index}.jpg', img_with_mesh)

        # call reconstructionError
        error = eval.reconstructionError(optimized_jt, gt_jt)
        errors.append(error)
        _LOGGER.info(f'reconstruction error for image {index} = {error}')

        index += 1
    
    errors = np.asarray(errors)
    avg_error = eval.avgReconError(errors)
    _LOGGER.info(f'average reconstruction error: {avg_error}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        'base_dir',
        default='../',
        nargs='?',
        help="Directory that contains images/lsp and results/lps , i.e."
        "the directory you untared smplify_code.tar.gz")
    parser.add_argument(
        '--out_dir',
        default='../pkl_output',
        type=str,
        help='Where results will be saved, default is pkl_output')
    parser.add_argument(
        '--no_interpenetration',
        default=False,
        action='store_true',
        help="Using this flag removes the interpenetration term, which speeds"
        "up optimization at the expense of possible interpenetration.")
    parser.add_argument(
        '--gender_neutral',
        default=False,
        action='store_true',
        help="Using this flag always uses the neutral SMPL model, otherwise "
        "gender specified SMPL models are used.")
    parser.add_argument(
        '--n_betas',
        default=10,
        type=int,
        help="Specify the number of shape coefficients to use.")
    parser.add_argument(
        '--flength',
        default=5000,
        type=float,
        help="Specify value of focal length.")
    parser.add_argument(
        '--side_view_thsh',
        default=25,
        type=float,
        help="This is thresholding value that determines whether the human is captured in a side view. If the pixel distance between the shoulders is less than this value, two initializations of SMPL fits are tried.")
    parser.add_argument(
        '--viz',
        default=True,
        action='store_true',
        help="Turns on visualization of intermediate optimization steps "
        "and final results.")
    parser.add_argument(
        '--no_pose',
        default=False,
        action='store_true',
        help="Using this flag removes the pose prior.")
    parser.add_argument(
        '--no_shape',
        default=False,
        action='store_true',
        help="Using this flag removes the shape prior.")
    args = parser.parse_args()

    use_interpenetration = not args.no_interpenetration
    if not use_interpenetration:
        _LOGGER.info('Not using interpenetration term.')
    if args.gender_neutral:
        _LOGGER.info('Using gender neutral model.')
    
    use_pose = not args.no_pose
    if not use_pose:
        _LOGGER.info('Not using pose.')

    use_shape = not args.no_shape
    if not use_shape:
        _LOGGER.info('Not using shape.')

    # Set up paths & load models.
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    main(args.base_dir, args.out_dir, use_interpenetration, args.n_betas,
         args.flength, args.side_view_thsh, args.gender_neutral, args.viz, use_pose, use_shape)