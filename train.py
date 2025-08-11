import numpy as np
import random
import os
import cv2
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, scale_loss, get_normal_smoothness, \
    cos_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.timer import Timer
from scene.external import get_add_point, MutiPlane_init, get_depth_mask, get_vector, error_map, gaussian_decomp

from time import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, pipe, testing_iterations, saving_iterations, sample_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer):
    first_iter = 0
    viewpoint_stack = None

    if checkpoint:
        print("restor gaussian")
        first_iter = checkpoint_iterations[0]
        gaussians.restore(os.path.join(dataset.source_path, dataset.model_path), checkpoint_iterations[0])

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter -= 1

    scale = 1
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras(scale).copy()[:]

    resample_num = 2

    for num in range(resample_num):
        idx = randint(0, len(viewpoint_stack) - 1)
        resample_cam = viewpoint_stack[idx]

        xyz = gaussians.get_xyz
        file_name = resample_cam.image_name[:-4]
        monodepth = resample_cam.depth
        init_xyz, init_features = MutiPlane_init(monodepth, xyz, resample_cam, plane_num=16,
                                                 sample_size=args.sample_win_size, muti_mode=args.muti_mode,
                                                 itera_num=num)
        gaussians.init_MutiPlane(init_xyzs=init_xyz, init_features=init_features)

    gaussians.training_setup(opt)

    viewpoint_stack = None
    for iteration in range(first_iter, final_iter + 1):

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        scale = 1

        # if iteration >= 5000:
        #     scale = 1
        # elif iteration >= 2000:
        #     scale = 2
        # elif iteration < 2000:
        #     scale = 4

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                                                                  render_pkg["visibility_filter"], render_pkg["radii"]
        render_depth = render_pkg["depth"]
        normal = render_pkg["normal"]
        # mask_vis = torch.zeros_like(depth)

        opac = render_pkg["opac"]
        mask_vis = (opac.detach() > 1e-5)

        d2n = depth2normal(render_depth, mask_vis, viewpoint_cam)

        gt_image = viewpoint_cam.original_image.cuda()

        losses = {}

        losses['image'] = l1_loss(image, gt_image)

        psnr_ = psnr(image, gt_image).mean().double()

        if opt.lambda_dssim != 0 and iteration > opt.densify_until_iter:
            losses['simm'] = 1.0 - ssim(image, gt_image)
            # loss += opt.lambda_dssim * (1.0-ssim_loss)

        if opt.lambda_scale_anc != 0:
            losses['scale'] = scale_loss(gaussians.get_scaling, therd_r=torch.Tensor([10.0]).to("cuda"))

        # print(normals_tensor.shape,gt_image_tensor.shape)
        # exit()

        if iteration < 3000:
            lambda_normal = 0.01
            lambda_normal_local = 0.001
        else:
            lambda_normal = 0.0001
            lambda_normal_local = 0.0001

        losses['normal'] = cos_loss(d2n, normal)

        losses['local_nomal'] = get_normal_smoothness(normal, gt_image, k_size=3)

        loss_weights = {'image': 1.0, 'simm': opt.lambda_dssim, 'scale': opt.lambda_scale_anc, "normal": lambda_normal,
                        "local_nomal": lambda_normal_local}

        loss = sum([loss_weights[k] * v for k, v in losses.items()])

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            # normal_loss_for_log = 0.4 *loss["normal"].item()+ 0.6 * normal_loss_for_log
            total_point = gaussians._xyz.shape[0]

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          # "normal":f"{normal_loss_for_log:.{4}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            training_report(tb_writer, iteration, losses['image'], loss, l1_loss, testing_iterations, scene, render,
                            [pipe, background])

            # Densification

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                opacity_threshold = opt.opacity_threshold
                densify_threshold = opt.densify_grad_threshold

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 100 if iteration > opt.opacity_reset_interval else None

                    gaussians.rescale(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration >= 1000 and iteration <= 3000 and iteration % 1000 == 0:
                    gaussians.prune_point_ours_small(num=args.prune_num1, std=args.prune_std1, planer_numer=16)
                elif iteration > 3000 and iteration % 2000 == 0 and iteration < opt.densify_until_iter - 2000:
                    gaussians.prune_point_ours_small(num=args.prune_num2, std=args.prune_std2, planer_numer=16)

            # Optimizer step

            if iteration in sample_iterations:
                print("resample !!  ", iteration)
                Gaussian_mask = np.zeros(gaussians.get_xyz.shape[0])
                masks = []
                val_cams = []
                for num in range(resample_num):
                    idx = randint(0, len(viewpoint_stack) - 1)
                    val_cams.append(viewpoint_stack[idx])

                for index, viewpoint_cam in enumerate(val_cams):
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                        "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                    render_depth = render_pkg["depth"]
                    mono_depth = viewpoint_cam.depth
                    if mono_depth.shape != render_depth.shape:
                        mono_depth = cv2.resize(mono_depth, [render_depth.shape[2], render_depth.shape[1]])

                    mask = get_depth_mask(render_depth.detach().cpu().numpy()[0], mono_depth)

                    error_mask = error_map(image, viewpoint_cam.original_image, static_factor=0.2)
                    mask[error_mask.detach().cpu().numpy() != 0] = 1
                    masks.append(mask)
                    Gaussian_mask_t = get_add_point(viewpoint_cam, gaussians.get_xyz, mask)

                    Gaussian_mask = Gaussian_mask + Gaussian_mask_t

                Gaussian_mask[Gaussian_mask < 2] = 0

                decomp_mask = Gaussian_mask[Gaussian_mask != 0]
                decomp_mask = torch.tensor(decomp_mask, dtype=torch.long)
                for i, view in enumerate(val_cams):
                    input_mask = masks[i]
                    gaussian_decomp(gaussians, view, input_mask, decomp_mask.to('cuda'))

                xyzs = gaussians.get_xyz[Gaussian_mask != 0]
                max_depth = gaussians.get_xyz.max() * 0.95
                win_sizes = [1, 7, 21]
                xyz_vector = get_vector(val_cams, xyzs, max_depth, split_num=5, win_sizes=win_sizes)

                gaussians.reset_xyz(xyz_vector, decomp_mask)

                ill_scale_mask = gaussians.get_scaling.min(dim=1).values < 1e-8
                false_count = (ill_scale_mask == True).sum()
                if false_count != 0:
                    print(iteration, false_count)
                    gaussians.removeScale(ill_scale_mask)

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, sample_iterations, checkpoint_iterations,
             checkpoint, debug_from, expname, vial_view):
    # first_iter = 0

    tb_writer = prepare_output_and_logger(expname)

    gaussians = GaussianModel(dataset.sh_degree)

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, vial_view=vial_view, load_coarse=None, resolution_scales=[1, 2, 4])
    timer.start()

    scene_reconstruction(dataset, opt, pipe, testing_iterations, saving_iterations, sample_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, args.iterations, timer)


def prepare_output_and_logger(expname):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(os.path.join(args.source_path, args.model_path)))
    os.makedirs(os.path.join(args.source_path, args.model_path), exist_ok=True)
    with open(os.path.join(os.path.join(args.source_path, args.model_path), "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.source_path, args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test',
                               'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                           range(10, 5000, 299)]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 1000 for i in range(0, 120)])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[3000, 5000, 10000, 30_000, 45000, 60000])
    parser.add_argument("--sample_iterations", nargs="+", type=int, default=[3000, 9000, 13000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="./output/test")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    args.model_path = args.model_path +"_"+args.muti_mode

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,args.sample_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname,args.vial_view)

    # All done
    print("\nTraining complete.")


