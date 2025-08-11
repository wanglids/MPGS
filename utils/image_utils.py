import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
@torch.no_grad()
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


from utils.graphics_utils import fov2focal, focal2fov
def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]),
                             indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:, :-2, 1:-1, :] - p_c) * mask[:, :-2, 1:-1, :]
    p_l = (p[:, 1:-1, :-2, :] - p_c) * mask[:, 1:-1, :-2, :]
    p_b = (p[:, 2:, 1:-1, :] - p_c) * mask[:, 2:, 1:-1, :]
    p_r = (p[:, 1:-1, 2:, :] - p_c) * mask[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    mask = mask[0, 1:-1, 1:-1, :]

    n = torch.nn.functional.normalize(n, dim=-1)

    n = (n * mask).permute([2, 0, 1])
    return n


def depth2wpos(depth, mask, camera):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]),
                             indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = torch.inverse(K).to(device)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    pose = camera.world_view_transform.to(device)
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    camWPos = (camPos - t) @ Rinv.t()

    camWPos = (camWPos[..., :3] * mask).permute([2, 0, 1])

    return camWPos

def resample_points(camera, depth, normal, color, mask):
    camWPos = depth2wpos(depth, mask, camera).permute([1, 2, 0])
    camN = normal.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0]).to(torch.bool)
    mask = mask.detach()[..., 0]
    camN = camN.detach()[mask]
    camWPos = camWPos.detach()[mask]
    camRGB = color.permute([1, 2, 0])[mask]

    Rinv = camera.world_view_transform[:3, :3].cuda()

    # print(camWPos.device, camN.device, Rinv.device, camRGB.device)
    points = torch.cat([camWPos, camN @ Rinv.t(), camRGB], -1)
    return points

def grid_prune(grid, shift, scale, dim, pts, thrsh=1):
    # print(dim)
    grid_cord = ((pts + shift) * scale).to(torch.long)
    # print(grid_cord.min(), grid_cord.max())
    out = (torch.le(grid_cord, 0) + torch.gt(grid_cord, dim - 1)).any(1)
    # print(grid_cord.min(), grid_cord.max())
    grid_cord = grid_cord.clamp(torch.zeros_like(dim), dim - 1)
    mask = grid[grid_cord[:, 0], grid_cord[:, 1], grid_cord[:, 2]] > thrsh
    mask *= ~out
    # print(grid_cord.shape, mask.shape, mask.sum())
    return mask.to(torch.bool)
