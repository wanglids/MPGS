
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussian_renderer import GaussianModel
from time import time
import lpips
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

from utils.image_utils import depth2normal, resample_points,grid_prune
from utils.general_utils import safe_state, poisson_mesh

from utils.loss_utils import calc_ssim,calc_psnr
device = torch.device("cuda:0")


def depth_to_pseudo_color(depth_image):
    # 将深度图转换为浮点型，进行归一化处理，然后转换回8位图像
    depth_image_32f = (depth_image-depth_image.min())/(depth_image.max()-depth_image.min())
    # depth_image_32f = 1-depth_image_32f
    pseudo_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_32f, alpha=255), cv2.COLORMAP_JET)
    return pseudo_color_image



def render_set(source_path,model_path, name, iteration, views, gaussians, pipeline, background,dataname,scenename,poisson_depth):
    render_path = os.path.join(source_path,model_path, name,"renders")
    gts_path = os.path.join(source_path,model_path, name, "gt")

    render_depth_path = os.path.join(source_path, model_path, name, "renders_depth")

    model_name = model_path.split("/")[1]
    save_path = f"E:\dataset_result/render_quailty/{model_name}"
    makedirs(save_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_depth_path,exist_ok=True)


    resampled = []

    bound = None
    occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, 512, bound)

    all_time = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        depth_range = [0, 20]
        mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])
        mask_vis = torch.ones_like(depth, device="cuda")
        normal = depth2normal(depth, mask_vis, view)
        # print(depth.device,normal.device,rendering.device,mask_clip.device)
        pts = resample_points(view, depth, normal, rendering, mask_clip)
        grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1)
        clean_mask = grid_mask  # * mask_mask
        pts = pts[clean_mask]
        resampled.append(pts.cpu())

    resampled = torch.cat(resampled, 0)
    mesh_path = f'{source_path}/{model_path}/poisson_mesh_{poisson_depth}'

    poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 3 * 1e-5)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,vial_view, skip_train : bool, skip_test : bool, skip_video: bool,dataname,scenename):
    with torch.no_grad():
        
        iteration = 30000
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians,vial_view= vial_view, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        poisson_depth = 10
        render_set(dataset.source_path, dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(),
                   gaussians, pipeline, background, dataname, scenename,poisson_depth)


        # if not skip_test:
        #     render_set(dataset.source_path,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,dataname,scenename)
        # if not skip_video:
        #     render_set(dataset.source_path,dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,dataname,scenename)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args()


    dir_path = args.source_path
    model_name = args.model_path

    datanames = os.listdir(dir_path)
    from time import gmtime

    fail_scene = ["scene_006","scene_008","scene_011","scene_038", "scene_083"]
    for dataname in datanames:
        if dataname == "ICL-NUIM":
            continue
        # if dataname!="human":
        #     continue
        scenenames = os.listdir(os.path.join(dir_path, dataname))


        for scenename in scenenames:

            if scenename in fail_scene:
                print(scenename, " is fail to reconsturct!!!!")
                continue
            print(dataname,scenename)

            args = get_combined_args(parser,dataname,scenename)

            args.source_path = os.path.join(dir_path, dataname, scenename)

            args.model_path = f"output/{model_name}"

            args.configs = f"./arguments/{dataname}/{scenename}.py"

            mesh_path = f'{args.source_path}/{args.model_path}/poisson_mesh_10_plain.ply'
            if os.path.exists(mesh_path):
                print(dataname, scenename, "has been create mesh!!!!")
                continue

            print("#######################")
            print("Rendering ", dataname, scenename)
            print("#######################")

            print("Rendering ", args.model_path)
            if args.configs:
                import mmengine
                from utils.params_utils import merge_hparams
                config = mmengine.Config.fromfile(args.configs)
                args = merge_hparams(args, config)
                print("Rendering " + args.model_path)
            # Initialize system state (RNG)


            safe_state(args.quiet)
            print(args.vial_view)
            render_sets(model.extract(args), args.iteration, pipeline.extract(args),args.vial_view, args.skip_train, args.skip_test, args.skip_video,dataname,scenename)