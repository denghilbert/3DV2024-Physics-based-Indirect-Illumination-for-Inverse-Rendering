import os
import torch
import numpy as np
import sys
sys.path.append("..")
import utils.general as utils
from utils import rend_util
import json
from datasets.read_colmap import calcu_3d_bbox


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.5 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        c2w[2, 3] = radius * 0.5
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 6, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def process_poses(poses, bounds, pts_file):
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    poses, pose_avg = center_poses(poses)
    # center_poses

    bbox = calcu_3d_bbox(pts_file)
    pts_r, pts_c = bbox['pts_r'], bbox['pts_c']
    pts_r = pts_r * 1.5
    # translation
    last_row = np.ones(1)
    pts_c = pose_avg @ np.concatenate([pts_c, last_row])
    poses[..., 3] -= pts_c[None, :3].repeat(poses.shape[0], axis=0)
    # scale
    poses[..., 3] /= pts_r
    bounds = bounds / pts_r
    print("Scale {}".format(pts_r))

    return poses, bounds


class RealDataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train',
                 train_cameras=False,
                 train_skip=[]
                 ):
        self.instance_dir = instance_dir
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = 2.2
        self.train_cameras = train_cameras
        self.split = split

        image_dir = os.path.join(self.instance_dir, 'image')
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = os.path.join(self.instance_dir, 'mask')
        mask_paths = sorted(utils.glob_imgs(mask_dir))
        poses_bounds = np.load(os.path.join(self.instance_dir, 'poses_bounds.npy'))
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]
        img_h, img_w, focal = poses[0, :, -1]
        print("focal {}, img_w {}, img_h {}".format(focal, img_w, img_h))
        img_h, img_w = np.int(img_h), np.int(img_w)

        poses, self.bounds = process_poses(poses, self.bounds, os.path.join(self.instance_dir, 'points3D.bin'))

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        if split == 'train' or split == 'val':
            # split dataset
            interval_ = len(image_paths) / 100.
            indices = [int(i) for i in list(np.arange(0, len(image_paths) - 1, interval_))]
            indices = [i for i in indices if i not in train_skip]
            # print(indices)
            print(f'Images\' number used for train is {len(indices)}')

            image_paths = np.array(image_paths)[indices]
            mask_paths = np.array(mask_paths)[indices]
            poses = np.array(poses)[indices]

            self.n_cameras = image_paths.shape[0]

            self.intrinsics_all = []
            self.pose_all = []

            intrinsics = [[focal, 0, img_w / 2], [0, focal, img_h / 2], [0, 0, 1]]
            intrinsics = np.array(intrinsics).astype(np.float32)

            for i in range(self.n_cameras):
                pose = poses[i, :3, :4]
                pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], 0)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.has_groundtruth = True
            self.rgb_images = []
            self.object_masks = []
            print('Applying inverse gamma correction: ', self.gamma)
            import pdb
            pdb.set_trace()
            for i in range(self.n_cameras):
                rgb, object_mask = rend_util.load_rgb_colmap(image_paths[i])
                # rgb = np.power(rgb, self.gamma)

                H, W = rgb.shape[1:3]
                self.img_res = [H, W]
                self.total_pixels = self.img_res[0] * self.img_res[1]


                rgb = rgb.reshape(-1, 3)
                self.rgb_images.append(torch.from_numpy(rgb).float())
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())


        elif split == 'test':
            self.has_groundtruth = False
            print('No ground-truth images available. Image resolution: ', img_w, img_w)
            self.img_res = [img_w, img_w]
            self.total_pixels = np.int(self.img_res[0] * self.img_res[1])

            radius = 2.0  # 1.5/0.8
            print(radius)
            pose_test = create_spheric_poses(radius)
            self.n_cameras = pose_test.shape[0]

            self.pose_all = []
            self.intrinsics_all = []
            intrinsics = [[focal, 0, img_w / 2], [0, focal, img_w / 2], [0, 0, 1]]
            intrinsics = np.array(intrinsics).astype(np.float32)
            for i in range(self.n_cameras):
                pose = pose_test[i, :3, :4]
                pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], 0)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            self.rgb_images = torch.ones([self.n_cameras, 3, self.total_pixels], dtype=torch.float32)
            self.object_masks = torch.ones([self.n_cameras, self.total_pixels]).bool()

    def __len__(self):
        return self.n_cameras

    def return_single_img(self, img_name):
        self.single_imgname = img_name
        for idx in range(len(self.image_paths)):
            if os.path.basename(self.image_paths[idx]) == self.single_imgname:
                self.single_imgname_idx = idx
                break
        print('Always return: ', self.single_imgname, self.single_imgname_idx)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

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