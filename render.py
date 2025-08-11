
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
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import lpips
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


from utils.loss_utils import calc_ssim,calc_psnr
device = torch.device("cuda:0")


def render_set(source_path,model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(source_path,model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(source_path,model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    PSNR = 0
    SSIM = 0
    LPIPS = 0
    lpips_vgg = lpips.LPIPS(net="vgg").cuda()


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if name !="video":
            SSIM += calc_ssim(rendering.unsqueeze(0), view.original_image.unsqueeze(0))
            PSNR += calc_psnr(rendering, view.original_image).mean().double()
            LPIPS += lpips_vgg(rendering.unsqueeze(0).to(device=device), view.original_image.unsqueeze(0).to(device=device))

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    count = 0
    # exit()
    print("writing training images.")
    if name !="video":
        SSIM = SSIM / len(views)
        PSNR = PSNR / len(views)
        LPIPS = LPIPS / len(views)
        output_text = f"{name}    SIMM: {SSIM}, PSNR: {PSNR}, LPIPS: {LPIPS}"
        print(output_text)
        text_path = os.path.dirname(render_path) + f"/{name}_SIMM_PSNR.txt"

        with open(text_path, "w") as f:
            f.write(output_text)

    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    if len(render_images)!=0:
        imageio.mimwrite(os.path.join(source_path,model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams,vial_view, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        
        iteration = 30000
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians,vial_view= vial_view, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test:
            render_set(dataset.source_path,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_video:
            render_set(dataset.source_path,dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # print(args.vial_view)
    # exit()

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),args.vial_view, args.skip_train, args.skip_test, args.skip_video)