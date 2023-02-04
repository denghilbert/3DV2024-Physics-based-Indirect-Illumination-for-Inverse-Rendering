import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import json
import imageio


class DTUDataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train'
                 ):
        self.split = split
        self.instance_dir = instance_dir
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        image_dir = os.path.join(self.instance_dir, 'image')
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = os.path.join(self.instance_dir, 'mask')
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.gamma = 2.2
        rgb_ = rend_util.load_rgb(image_paths[0])
        rgb_ = np.power(rgb_, self.gamma)
        H, W = rgb_.shape[:2]
        self.img_res = [H, W]
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.n_images = len(image_paths)
        self.n_cameras = len(image_paths)

        self.cam_file = os.path.join(self.instance_dir, 'cameras.npz')


        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]


        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics[:3, :3]).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb_DTU(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())


        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def __len__(self):
        return (self.n_cameras)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "object_mask": self.object_masks[idx],
        }
        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input,
        # ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]


