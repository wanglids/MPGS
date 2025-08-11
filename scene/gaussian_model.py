
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import open3d as o3d
from torch.utils.cpp_extension import load

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        print("start load cuda")
        self.utils_mod = load(name="cuda_utils", sources=["./utils/ext.cpp", "./utils/cuda_utils.cu"])
        print("end load cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, checkpoint_iterations):
        self.load_ply(os.path.join(model_args,f'point_cloud/iteration_{checkpoint_iterations}/point_cloud.ply'))

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def init_MutiPlane(self,init_xyzs,init_features):

        self._features_dc = nn.Parameter(torch.cat((self._features_dc,init_features.unsqueeze(1)),dim=0).contiguous().requires_grad_(True))
        self._features_rest  = nn.Parameter(torch.cat((self._features_rest,torch.zeros((init_features.shape[0],((self.max_sh_degree + 1) ** 2)-1,init_features.shape[1]),device="cuda")),dim=0).contiguous().requires_grad_(True))
        self._xyz = nn.Parameter(torch.cat((self._xyz,init_xyzs),dim=0).contiguous().requires_grad_(True))
        dist2 = torch.clamp_min(distCUDA2(self._xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        opacities = inverse_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def add_MutiPlane(self,init_xyzs,init_features):

        self._features_dc = nn.Parameter(init_features.unsqueeze(1).contiguous().requires_grad_(True))
        self._features_rest  = nn.Parameter(torch.zeros((init_features.shape[0],((self.max_sh_degree + 1) ** 2)-1,init_features.shape[1]),device="cuda").contiguous().requires_grad_(True))
        self._xyz = nn.Parameter(init_xyzs.contiguous().requires_grad_(True))
        dist2 = torch.clamp_min(distCUDA2(self._xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        opacities = inverse_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def restore_xyz(self,xyzs,gaussian_mask):
        self._xyz[gaussian_mask] = xyzs

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_select_ply(self, path,Gaussian_mask):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]


        xyz = xyz[Gaussian_mask!=0]
        normals = normals[Gaussian_mask != 0]
        f_dc = f_dc[Gaussian_mask != 0]
        f_rest = f_rest[Gaussian_mask != 0]
        opacities = opacities[Gaussian_mask != 0]
        scale = scale[Gaussian_mask != 0]
        rotation = rotation[Gaussian_mask != 0]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_select_xyz_ply(self, path,Gaussian_mask,xyzs):
        mkdir_p(os.path.dirname(path))

        xyz = xyzs.detach().cpu().numpy()

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        normals = np.zeros_like(scale)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        normals = normals[Gaussian_mask != 0]
        f_dc = f_dc[Gaussian_mask != 0]
        f_rest = f_rest[Gaussian_mask != 0]
        opacities = opacities[Gaussian_mask != 0]
        scale = scale[Gaussian_mask != 0]
        rotation = rotation[Gaussian_mask != 0]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")

        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]




    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def split_boundry(self,xyz,scale,rotation,scale_factor,dist_vecs):

        split_num = len(scale_factor)
        scale_factor = torch.tensor(np.array(scale_factor),device='cuda')
        dist_vecs = torch.tensor(np.array(dist_vecs),device="cuda")
        stds = scale
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation)
        xyz_vector = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # new_scale = torch.zeros(( split_num,scale.shape[1]),device="cuda")
        new_scale = scale*scale_factor
        new_rotation = rotation

        new_xyz = xyz+(1-scale_factor)*xyz_vector*dist_vecs

        return new_xyz,new_scale,new_rotation



    # def split_boundry(self,xyz,scale,rotation,scale_factor):
    #     split_num = len(scale_factor)
    #     scale_factor = torch.tensor(scale_factor,device='cuda')
    #     stds = scale
    #     means = torch.zeros((stds.size(0), 3), device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(rotation)
    #     xyz_vector = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
    #     # xyz_vector[xyz_vector!=torch.max(xyz_vector)] = 0
    #     # unit_vector = torch.nn.functional.normalize(xyz_vector)
    #     # print("unit_vector",unit_vector)
    #     print("xyz",xyz)
    #     print("scale",scale)
    #     print("xyz_vector",xyz_vector)
    #     new_scale = torch.zeros((split_num,scale.shape[1]),device="cuda")
    #     new_xyz = torch.zeros((split_num, xyz.shape[1]), device="cuda")
    #     new_rotation = torch.zeros((split_num, rotation.shape[1]), device="cuda")
    #     for i in range(split_num):
    #         new_scale[i] = scale*scale_factor[i]
    #         new_rotation[i] = rotation
    #         print("scale_factor",scale_factor,0.5*(scale-scale_factor[i]*scale))
    #         # new_xyz[i] = xyz+0.5*(1-scale_factor[i])*(unit_vector)*((-1)**(i))
    #         # new_xyz[i] = xyz+0.5*(1-scale_factor[i])*(xyz_vector/scale)*((-1)**(i))
    #         new_xyz[i] = xyz + (1 - scale_factor[i]) * (xyz_vector) * ((-1) ** (i))
    #         print("i", (-1) ** (i + 1), new_xyz[i])
    #
    #     return new_xyz,new_scale,new_rotation

        # unit_xyz_vector = xyz_vector/
        # new_xyz =  + xyz.repeat(N, 1)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    def add_point(self,new_point,near_point_mask):

        repeat_num = []
        selected_pts_mask = []
        for i in range(near_point_mask.shape[0]):
            if near_point_mask[i] in selected_pts_mask:
                continue
            cover = torch.arange(near_point_mask.size(0))
            idx = cover[near_point_mask == near_point_mask[i]]
            repeat_num.append(len(idx))
            selected_pts_mask.append(near_point_mask[i])
        
        repeat_num = torch.tensor(repeat_num,device="cuda")
        selected_pts_mask = torch.tensor(selected_pts_mask,device="cuda")

        stds = self.get_scaling[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat_interleave(repeat_num+1, dim=0)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)/ (0.8*(torch.mean(repeat_num.float())+1)))
        new_rotation = self._rotation[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat_interleave(repeat_num+1, dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        #  densify_grad_threshold_fine_init 0.0002; scene_extent camera L2 normal
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    def rescale(self, max_grad, min_opacity, extent, max_screen_size):


        if max_screen_size:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            self.rescale_point(big_points_ws)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # ill_scale_mask = self.get_scaling.min(dim=1).values <0.00001
        # prune_mask = torch.logical_or(prune_mask, ill_scale_mask)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def removeScale(self,ill_scale_mask):
        self.prune_points(ill_scale_mask)
        torch.cuda.empty_cache()

    def prune(self,max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def rescale_point(self,selected_pts_mask):
        self._scaling[selected_pts_mask] = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask]/2)


    def split_point(self,selected_pts_mask):
        N = 2

        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)




    def prune_point_ours(self,stage):
        pcd_vector = o3d.geometry.PointCloud()
        pcd_vector.points = o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
        mask_point_temp = torch.logical_or(torch.ones(self._xyz.shape[0]),torch.ones(self._xyz.shape[0]))
        if stage == "coarse":
            point_temp = np.array(pcd_vector.points)
            z_min = point_temp[:,2].min()
            z_mean = np.mean(point_temp[:,2])
            thrd_z = (z_mean-z_min)*0.6
            mask_point = point_temp[:,2]>(thrd_z+z_min)
            
            mask_point_temp = torch.logical_and(mask_point_temp,torch.tensor(mask_point))

            pp = point_temp[mask_point]
            pcd_vector = o3d.geometry.PointCloud()
            pcd_vector.points = o3d.utility.Vector3dVector(pp)
        if stage == "coarse":
            cl, ind1 = pcd_vector.remove_statistical_outlier(500, 0.5)
            pcd_vector = pcd_vector.select_by_index(ind1)
            cl, ind2 = pcd_vector.remove_statistical_outlier(500, 0.5)
            pcd_vector = pcd_vector.select_by_index(ind2)
            cl, ind3 = pcd_vector.remove_statistical_outlier(1000, 0.5)
            ind = list(np.array(ind1)[np.array(ind2)[np.array(ind3)]])
        # cl, ind = pcd_vector.remove_statistical_outlier(20, 1)
        else:
            cl, ind = pcd_vector.remove_statistical_outlier(500, 0.5)
        mask_point_temp_1 = torch.logical_or(torch.zeros(self._xyz.shape[0]),torch.zeros(self._xyz.shape[0]))
        mask_point_temp_1[ind] = True
        mask_point_temp = torch.logical_and(mask_point_temp,mask_point_temp_1)

        self.prune_points(~mask_point_temp)

        torch.cuda.empty_cache()
    def remove_nan(self):
        xyz_t = torch.isnan(self.get_xyz)
        xyz_m = torch.logical_or(xyz_t[:,0],torch.logical_or(xyz_t[:,1],xyz_t[:,2]))

        scale_t = torch.isnan(self.get_scaling)
        scale_m = torch.logical_or(scale_t[:,0],torch.logical_or(scale_t[:,1],scale_t[:,2]))

        rotation_t = torch.isnan(self.get_rotation)
        rotation_m = torch.logical_or(rotation_t[:,0],torch.logical_or(rotation_t[:,1],torch.logical_or(rotation_t[:,2],rotation_t[:,3])))

        remove_mask = torch.logical_or(xyz_m,torch.logical_or(scale_m,rotation_m)) # nan True

        self.prune_points(remove_mask)

        torch.cuda.empty_cache()


    def prune_point_ours_small(self,num,std,planer_numer=None):
        if planer_numer == None:
            pcd_vector = o3d.geometry.PointCloud()
            pcd_vector.points = o3d.utility.Vector3dVector(self.get_xyz.detach().cpu().numpy())
            cl, ind = pcd_vector.remove_statistical_outlier(num, std)
            mask_point_temp = torch.logical_or(torch.zeros(self._xyz.shape[0]), torch.zeros(self._xyz.shape[0]))
            mask_point_temp[ind] = True
            self.prune_points(~mask_point_temp)
            torch.cuda.empty_cache()

        else:
            depth_min = self.get_xyz[:,2].min()
            depth_max = self.get_xyz[:,2].max()
            depth_vector = (depth_max-depth_min)/planer_numer
            depth_mask = torch.zeros(self.get_xyz.shape[0])

            for i in range(planer_numer):
                depth_mask[self.get_xyz[:, 2] <= (depth_min + (1 + i) * depth_vector)] += 1

            mask_point_temp = torch.logical_or(torch.zeros(self.get_xyz.shape[0]), torch.zeros(self.get_xyz.shape[0]))

            for i in range(planer_numer):
                muti_mask = torch.zeros_like(depth_mask)
                muti_mask[depth_mask == (i + 1)] = 1
                if torch.sum(muti_mask) < num * 10:
                    continue
                pcd_vector = o3d.geometry.PointCloud()
                pcd_vector.points = o3d.utility.Vector3dVector(self.get_xyz[muti_mask==1].detach().cpu().numpy())
                cl, ind = pcd_vector.remove_statistical_outlier(num, std)
                mask_t = torch.zeros(muti_mask[muti_mask==1].shape[0])
                mask_t[ind] = 1
                muti_mask[muti_mask==1] = mask_t
                mask_point_temp[muti_mask==1] = True

            self.prune_points(~mask_point_temp)


            torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    def reset_xyz(self,xyz,mask):
        self._xyz[mask] = nn.Parameter(xyz).contiguous().requires_grad_(True)

    def reset_scaling(self,scale,mask):
        self._scaling[mask] = nn.Parameter(self.scaling_inverse_activation(scale)).contiguous().requires_grad_(True)


    def reset_xyz_color(self,xyz,color):

        fused_color = RGB2SH(torch.tensor(np.asarray(color)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_color.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        xyz = torch.tensor(xyz,dtype=torch.float, device="cuda")
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        # print(self.get_xyz.shape)


        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # 100*3*1
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # 100*3*15
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # print(xyz_min, xyz_max, xyz_len)

        # grid_dim_max = 1024
        grid_len = xyz_len.max() / grid_dim_max
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               self.get_xyz, self.get_rotation, self.get_scaling, self.get_opacity,
                                               torch.tensor([cutoff]).to(torch.float32).cuda())

        # print('here')
        # x, y, z = torch.meshgrid(torch.arange(0, grid_dim[0]), torch.arange(0, grid_dim[1]), torch.arange(0, grid_dim[2]), indexing='ij')

        # print('here')
        # exit()

        # grid_cord = torch.stack([x, y, z], -1).cuda()

        return grid, -xyz_min, 1 / grid_len, grid_dim