import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

from tqdm import tqdm
import pymeshlab
from pytorch3d.ops import knn_points

def poisson_mesh(path, vtx, normal, color, depth, thrsh):
    pbar = tqdm(total=4)
    pbar.update(1)
    pbar.set_description('Poisson meshing')

    # create pcl with normal from sampled points
    ms = pymeshlab.MeshSet()
    pts = pymeshlab.Mesh(vtx.cpu().numpy(), [], normal.cpu().numpy())
    ms.add_mesh(pts)

    # poisson reconstruction
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True, samplespernode=1.5)
    vert = ms.current_mesh().vertex_matrix()
    face = ms.current_mesh().face_matrix()
    ms.save_current_mesh(path + '_plain.ply')

    pbar.update(1)
    pbar.set_description('Mesh refining')
    # knn to compute distance and color of poisson-meshed points to sampled points
    nn_dist, nn_idx, _ = knn_points(torch.from_numpy(vert).to(torch.float32).cuda()[None], vtx.cuda()[None], K=4)
    nn_dist = nn_dist[0]
    nn_idx = nn_idx[0]
    nn_color = torch.mean(color[nn_idx], axis=1)

    # create mesh with color and quality (distance to the closest sampled points)
    vert_color = nn_color.clip(0, 1).cpu().numpy()
    vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
    ms.add_mesh(pymeshlab.Mesh(vert, face, v_color_matrix=vert_color, v_scalar_array=nn_dist[:, 0].cpu().numpy()))

    pbar.update(1)
    pbar.set_description('Mesh cleaning')
    # prune outlying vertices and faces in poisson mesh
    ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
    ms.meshing_remove_selected_vertices()

    # fill holes
    ms.meshing_close_holes(maxholesize=300)
    ms.save_current_mesh(path + '_pruned.ply')

    # smoothing, correct boundary aliasing due to pruning
    ms.load_new_mesh(path + '_pruned.ply')
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
    ms.save_current_mesh(path + '_pruned.ply')

    pbar.update(1)
    pbar.close()