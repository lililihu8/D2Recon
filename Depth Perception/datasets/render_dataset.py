import os
import os.path as osp
from functools import lru_cache

import numpy as np
import PIL.Image as pil

from .base_dataset import BaseDataset


# OpenCV camera convention -> Waymo/Dynamo dataset convention.
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


class RenderDataset(BaseDataset):
    """Dataset wrapper for rendered OmniRe sequences stored under Output/train|val."""

    def __init__(self, *args, **kwargs):
        super(RenderDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1920, 1280)
        self.cam_id = self._parse_cam_id(self.cam_name)
        self.has_mask_gt = False
        self.depth_is_dense = False
        self.pseudo_depth_min = 0.1
        self.pseudo_depth_max = 75.0

        self.K = {}
        self._frame_cache = {}
        self.get_all_intrinsic()

    @staticmethod
    def _parse_cam_id(cam_name):
        cam_map = {"FRONT": 0, "FRONT_LEFT": 1, "FRONT_RIGHT": 2, "SIDE_LEFT": 3, "SIDE_RIGHT": 4}
        if cam_name in cam_map:
            return cam_map[cam_name]
        try:
            return int(cam_name)
        except (TypeError, ValueError):
            return 0

    def get_timestep(self, folder, frame_index, offset):
        return 1

    def get_all_intrinsic(self):
        for file in self.filenames:
            folder = file.split()[0]
            if folder in self.K:
                continue

            intrinsic_path = osp.join(self.data_path, folder, "intrinsics", f"{self.cam_id}.txt")
            intrinsic = np.loadtxt(intrinsic_path).reshape(-1)
            fx, fy, cx, cy = intrinsic[:4]
            width, height = self.full_res_shape

            K = np.eye(4, dtype=np.float32)
            K[0, 0] = fx / width
            K[1, 1] = fy / height
            K[0, 2] = cx / width
            K[1, 2] = cy / height
            self.K[folder] = K

    def get_intrinsic(self, folder):
        return self.K[folder]

    def get_gt_dim(self, folder, frame_index, side):
        return self.full_res_shape[1], self.full_res_shape[0]

    def _rgb_dir(self, folder):
        base_path = osp.join(self.data_path, folder)
        gt_rgb_path = osp.join(base_path, "videos", "full_set_30000", "gt_rgbs")
        eval_rgb_path = osp.join(base_path, "videos_eval", "full_set_30000", "rgbs")
        normal_rgb_path = osp.join(base_path, "videos", "full_set_30000", "rgbs")
        # print(eval_rgb_path)
        if os.path.exists(gt_rgb_path):
            # print(eval_rgb_path)
            return gt_rgb_path
        return normal_rgb_path
        # return osp.join(self.data_path, folder, "videos", "full_set_30000", "rgbs")

    def _depth_dir(self, folder):
        return osp.join(self.data_path, folder, "videos", "full_set_30000", "lidar_depth")

    def get_img_path(self, folder, frame_index, side):
        return osp.join(self._rgb_dir(folder), f"{frame_index:03d}_000{self.img_ext}")

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_img_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    @staticmethod
    @lru_cache(maxsize=1)
    def _turbo_color_lut():
        # Approximate the common blue->red depth visualization with a coarse RGB cube LUT.
        turbo = np.array(
            [
                [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81], [53, 30, 88], [54, 33, 95],
                [55, 36, 102], [56, 39, 109], [57, 42, 115], [58, 45, 121], [59, 47, 128], [60, 50, 134],
                [61, 53, 139], [62, 56, 145], [63, 59, 150], [63, 62, 155], [64, 64, 159], [65, 67, 164],
                [66, 70, 168], [66, 73, 172], [67, 75, 176], [68, 78, 180], [68, 81, 183], [69, 83, 186],
                [69, 86, 189], [70, 89, 192], [70, 91, 195], [71, 94, 197], [71, 97, 199], [72, 99, 201],
                [72, 102, 203], [72, 104, 205], [73, 107, 207], [73, 109, 208], [74, 112, 210], [74, 114, 211],
                [74, 117, 212], [74, 119, 213], [75, 122, 214], [75, 124, 215], [75, 127, 216], [75, 129, 216],
                [75, 132, 217], [75, 134, 217], [76, 137, 218], [76, 139, 218], [76, 142, 218], [76, 144, 218],
                [76, 147, 218], [75, 149, 218], [75, 152, 218], [75, 154, 217], [75, 157, 217], [75, 159, 216],
                [74, 162, 216], [74, 164, 215], [74, 167, 214], [73, 169, 214], [73, 172, 213], [72, 174, 212],
                [72, 177, 211], [71, 179, 210], [71, 182, 208], [70, 184, 207], [69, 187, 206], [69, 189, 204],
                [68, 192, 203], [67, 194, 201], [66, 197, 199], [66, 199, 198], [65, 202, 196], [64, 204, 194],
                [63, 207, 192], [62, 209, 190], [61, 212, 188], [60, 214, 186], [59, 217, 184], [58, 219, 181],
                [57, 221, 179], [56, 224, 177], [55, 226, 174], [54, 229, 172], [53, 231, 169], [52, 233, 166],
                [51, 236, 163], [49, 238, 160], [48, 240, 157], [47, 243, 154], [46, 245, 151], [45, 247, 148],
                [44, 249, 145], [43, 251, 141], [42, 253, 138], [42, 255, 135], [43, 255, 131], [45, 255, 128],
                [48, 255, 124], [51, 255, 121], [54, 255, 117], [58, 255, 113], [62, 255, 109], [67, 255, 105],
                [72, 255, 101], [77, 255, 97], [82, 255, 93], [88, 254, 89], [94, 254, 85], [100, 254, 81],
                [106, 253, 77], [112, 253, 73], [118, 252, 69], [124, 251, 65], [131, 251, 61], [137, 250, 57],
                [143, 249, 53], [150, 248, 49], [156, 247, 46], [162, 246, 42], [168, 244, 38], [174, 243, 35],
                [180, 242, 31], [186, 240, 28], [192, 239, 25], [198, 237, 22], [204, 235, 19], [209, 234, 16],
                [215, 232, 14], [220, 230, 12], [226, 228, 10], [231, 226, 9], [236, 224, 8], [241, 221, 7],
                [246, 219, 7], [250, 216, 7], [254, 214, 8], [255, 211, 10], [255, 208, 13], [255, 205, 16],
                [255, 202, 20], [255, 199, 25], [255, 196, 30], [255, 193, 35], [255, 190, 41], [255, 186, 47],
                [255, 183, 53], [255, 179, 60], [255, 176, 67], [255, 172, 74], [255, 168, 81], [255, 164, 89],
                [255, 160, 97], [255, 156, 105], [255, 152, 113], [255, 147, 121], [255, 143, 130], [255, 138, 138],
                [255, 133, 147], [255, 128, 156], [255, 123, 165], [255, 118, 174], [255, 112, 183], [255, 107, 192],
                [255, 101, 201], [255, 95, 210], [255, 89, 219], [255, 83, 228], [255, 77, 237], [255, 71, 246],
                [255, 65, 255], [250, 59, 255], [244, 53, 255], [238, 47, 255], [232, 41, 255], [226, 35, 255],
                [220, 30, 255], [214, 25, 255], [207, 20, 255], [201, 16, 255], [195, 13, 255], [188, 10, 255],
                [182, 8, 255], [176, 7, 255], [169, 6, 255], [163, 6, 255], [157, 6, 255], [150, 8, 255],
                [144, 10, 255], [138, 13, 255], [132, 16, 255], [126, 20, 255], [120, 24, 255], [115, 29, 255],
                [109, 34, 255], [104, 40, 255], [99, 46, 255], [94, 52, 255], [90, 58, 255], [85, 64, 255],
                [81, 71, 255], [77, 78, 255], [74, 85, 255], [70, 92, 255], [67, 99, 255], [64, 106, 255],
                [61, 113, 255], [58, 120, 255], [56, 128, 255], [53, 135, 255], [51, 142, 255], [49, 149, 255],
                [47, 156, 255], [45, 163, 255], [43, 170, 255], [41, 176, 255], [39, 183, 255], [38, 189, 255],
                [36, 195, 255], [35, 201, 255], [33, 207, 255], [32, 213, 255], [31, 219, 255], [30, 224, 255],
                [29, 229, 255], [28, 234, 255], [27, 239, 255], [26, 243, 255], [25, 247, 255], [24, 251, 255],
                [23, 255, 255], [25, 255, 249], [29, 255, 240], [34, 255, 231], [40, 255, 221], [47, 255, 210],
                [55, 255, 198], [64, 255, 186], [74, 255, 173], [85, 255, 160], [97, 255, 146], [110, 255, 132],
                [124, 255, 118], [138, 255, 104], [153, 255, 90], [168, 255, 76],
            ],
            dtype=np.float32,
        )

        grid = np.stack(np.meshgrid(np.arange(32), np.arange(32), np.arange(32), indexing="ij"), axis=-1)
        colors = (grid.reshape(-1, 3).astype(np.float32) * 8.0) + 4.0
        dists = ((colors[:, None, :] - turbo[None, :, :]) ** 2).sum(-1)
        return np.argmin(dists, axis=1).astype(np.float32) / float(len(turbo) - 1)

    @classmethod
    def decode_colorized_depth(cls, depth_rgb, min_depth=0.1, max_depth=75.0):
        rgb = depth_rgb[..., :3].astype(np.uint8)
        quantized = (rgb >> 3).astype(np.int32)
        lut_idx = (quantized[..., 0] << 10) | (quantized[..., 1] << 5) | quantized[..., 2]
        scalar = cls._turbo_color_lut()[lut_idx]

        # Rendered depth maps typically use warm colors for close regions and cool colors for far regions.
        depth = min_depth + (1.0 - scalar) * (max_depth - min_depth)
        invalid = np.all(rgb == 0, axis=-1)
        depth[invalid] = 0.0
        return depth.astype(np.float32)

    def get_depths(self, folder, frame_index, side, do_flip):
        depth_path = osp.join(self._depth_dir(folder), f"{frame_index:03d}_000{self.img_ext}")
        depth_img = np.array(pil.open(depth_path))

        if depth_img.ndim == 2:
            depth = depth_img.astype(np.float32)
            if np.issubdtype(depth_img.dtype, np.integer):
                depth = depth / np.iinfo(depth_img.dtype).max * self.pseudo_depth_max
        else:
            depth = self.decode_colorized_depth(depth_img, self.pseudo_depth_min, self.pseudo_depth_max)

        if do_flip:
            depth = np.fliplr(depth)

        return depth
    
    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}{}".format(frame_index, '.npy')
        depth_path = osp.join(self._depth_dir(folder), f"{frame_index:06d}{'.npy'}")

        depth = np.load(depth_path)

        if do_flip:
            depth[:,0] = self.full_res_shape[0] - depth[:,0]
        
        depth = np.concatenate((depth[:,1:2], depth[:,0:1], depth[:,2:3]), axis=1)    # (N, 3) -> [h_i, w_i, z_i]

        return depth

    def get_mask(self, folder, frame_index, side, do_flip):
        shape = self.full_res_shape[::-1]
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)

    def get_frame_indices(self, folder):
        if folder not in self._frame_cache:
            frame_indices = []
            for fname in os.listdir(self._rgb_dir(folder)):
                stem, ext = osp.splitext(fname)
                if ext != self.img_ext:
                    continue
                frame_indices.append(int(stem.split("_")[0]))
            self._frame_cache[folder] = sorted(frame_indices)
        return self._frame_cache[folder]

    def get_odometry(self, folder):
        extrinsic = np.loadtxt(osp.join(self.data_path, folder, "extrinsics", f"{self.cam_id}.txt")).astype(np.float32)
        cam_to_ego = extrinsic @ OPENCV2DATASET

        poses = []
        for frame_idx in self.get_frame_indices(folder):
            ego_to_world = np.loadtxt(osp.join(self.data_path, folder, "ego_pose", f"{frame_idx:03d}.txt")).astype(np.float32)
            cam_to_world = ego_to_world @ cam_to_ego
            poses.append(cam_to_world)

        return np.stack(poses, axis=0)
