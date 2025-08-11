
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, vial_view = None,shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.source_path,self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path,vial_view)


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.source_path,self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)

            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.source_path,self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,args)
            self.video_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.video_cameras, resolution_scale,args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.source_path,self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):

        point_cloud_path = os.path.join(self.source_path,self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def save_select(self, iteration,cam_num,Gaussian_mask):

        point_cloud_path = os.path.join(self.source_path,self.model_path, "point_cloud/select_coarse_iteration_{}".format(iteration))
        self.gaussians.save_select_ply(os.path.join(point_cloud_path, f"point_cloud_select_{cam_num}.ply"),Gaussian_mask)

    def save_select_xyz(self, iteration,cam_num, stage,Gaussian_mask,xyzs):
        point_cloud_path = os.path.join(self.source_path,self.model_path, "point_cloud/select_coarse_iteration_{}".format(iteration))
        self.gaussians.save_select_xyz_ply(os.path.join(point_cloud_path, f"point_cloud_select_{cam_num}.ply"),Gaussian_mask,xyzs)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]