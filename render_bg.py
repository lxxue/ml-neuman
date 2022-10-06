#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render test views, and report the metrics
'''

import argparse
import os

import imageio
import lpips
import numpy as np
import skimage
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

loss_fn_alex = lpips.LPIPS(net='alex')

from data_io import neuman_helper
from models import human_nerf
from options import options
from utils import render_utils, utils


def eval_metrics(gts, preds):
    results = {
        'ssim': [],
        'psnr': [],
        'lpips': []
    }
    for gt, pred in zip(gts, preds):
        results['ssim'].append(ssim(pred, gt, multichannel=True))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred))
        results['lpips'].append(
            float(loss_fn_alex(utils.np_img_to_torch_img(pred[None])/127.5-1, utils.np_img_to_torch_img(gt[None])/127.5-1)[0, 0, 0, 0].data)
        )
    for k, v in results.items():
        results[k] = np.mean(v)
    return results


def optimize_pose_with_nerf(opt, cap, net, iters=1000, save_every=10):
    # TODO: optimize the pose with the trained NeRF
    pass


def main(opt):
    train_split, val_split, test_split = neuman_helper.create_split_files(opt.scene_dir)
    test_views = neuman_helper.read_text(test_split)
    val_views = neuman_helper.read_text(val_split)
    train_views = neuman_helper.read_text(train_split)
    # test_views = val_views + test_views
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.render_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale,
        smpl_type='optimized'
    )
    if opt.geo_threshold < 0:
        bones = []
        for i in range(len(scene.captures)):
            bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
        opt.geo_threshold = np.mean(bones)
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    for split_name, views in zip(["train", "val", "test"], [train_views, val_views, test_views]):
        all_world_pts = []
        all_colors = []
        save_path_dir = os.path.join('./bg_renders', f'{os.path.basename(opt.scene_dir)}/{split_name}_views')
        for view_name in tqdm(views):
            cap = scene[view_name]
            i = cap.frame_id['frame_id']
            rgb, depth_map, world_pts = render_utils.render_vanilla_depth(
                coarse_net=net.coarse_bkg_net,
                cap=cap,
                fine_net=net.fine_bkg_net,
                rays_per_batch=opt.rays_per_batch,
                samples_per_ray=opt.samples_per_ray,
                return_depth=True,
            )
            if not os.path.isdir(save_path_dir):
                os.makedirs(save_path_dir)
            imageio.imsave(os.path.join(save_path_dir, f'rgb_{str(i).zfill(4)}.png'), (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8))
            np.save(os.path.join(save_path_dir, f'depth_{i:04d}.npy'), depth_map)
            depth_color = (colorize(depth_map, 0.0, 3.0, cmap='turbo') * 255).astype(np.uint8)
            imageio.imsave(os.path.join(save_path_dir, f'depth_color_{i:04d}.png'), depth_color)

            save_points(os.path.join(save_path_dir, f'world_pts_{i:04d}.ply'), world_pts, colors=rgb.reshape(-1, 3), BRG2RGB=False)
            all_world_pts.append(world_pts)
            all_colors.append(rgb.reshape(-1, 3))
        all_world_pts = np.concatenate(all_world_pts, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        save_points(f"./bg_renders/{os.path.basename(opt.scene_dir)}/{split_name}_views/all_world_pts.ply", all_world_pts, colors=all_colors, BRG2RGB=False)

import open3d as o3d

from visualization import colorize


def save_points(path_save, pts, colors = None, normals = None, BRG2RGB=False):
    '''save points to point cloud using open3d
    '''
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--image_dir', required=False, type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')

    opt = parser.parse_args()
    assert opt.geo_threshold == -1, 'please use auto geo_threshold'
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)
    main(opt)
