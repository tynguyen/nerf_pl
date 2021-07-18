"""[Test LLFF dataloader and the data saved in poses_bounds.npy]
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
import os, glob
import open3d as o3d
from typing import overload
from PIL import Image
from datasets import LLFFDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
from utils.colmap_read_write import read_rgbd_images
from utils.colmap_read_write import getPCLfromRGB_D
import matplotlib.pyplot as plt

def show_pcl_list(pcl_list):
    # visualizer = o3d.visualization.Visualizer()
    # visualizer.create_window()

    # visualizer.add_geometries(pcl_list)
    # visualizer.destroy_window()
    o3d.visualization.draw_geometries(pcl_list)

class CustomLLFFDataset(LLFFDataset):
    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)

        image_names_file = os.path.join(self.root_dir,
                    'image_names_corresponding_to_poses_bounds.txt')
        with open(image_names_file, 'r') as fin:
            image_names = fin.readlines() # (N_images)
            image_names = [name.strip() for name in image_names]
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                        # load full resolution image then resize
        self.image_names = [name.split('/')[-1] for name in self.image_paths]
        # Make sure image_names and self.image_names are identical
        total_diffs = 0
        for i in range(len(image_names)):
            total_diffs += image_names[i] != self.image_names[i]
        assert total_diffs == 0 , "[Error] image_names list must be identical to self.image_names!"
        self.depth_paths = sorted(glob.glob(os.path.join(self.root_dir, 'depths/*')))
                        # load full resolution image then resize

        assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0]/W

        self.cv_intrinsics = np.array([
            [self.focal,    0,       self.img_wh[0]/2],
            [0         , self.focal, self.img_wh[1]/2],
            [0         ,    0,            0]])

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        self.poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal

    def get_pcls_from_rgbds(self):
        """Get pointclouds from RGB + Depth images
        """
        pcl_list = []
        rgbd_images = read_rgbd_images(self.root_dir, self.image_names, rgb_ext='.jpg', depth_ext='.png')

        for i, image_name in enumerate(self.image_names):
            rgb, depth = rgbd_images[image_name]
            # plt.subplot(121)
            # plt.imshow(depth, cmap="jet")
            depth = depth / 1000.0
            depth = depth.astype(np.float32)
            # plt.subplot(122)
            # plt.imshow(depth, cmap="jet")
            # plt.show()
            cvCam2W_T = self.poses[i] # 3x4
            cvCam2W_T = np.vstack([cvCam2W_T, np.array([0,0,0,1])])
            o3d_rgbd_pcl = getPCLfromRGB_D(rgb, depth, self.cv_intrinsics)

            # Transform this pcl
            o3d_rgbd_pcl.transform(cvCam2W_T)
            # pcl_list.append(o3d_rgbd_pcl)

            # Camera pose
            o3d_cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            o3d_cam_pose.transform(cvCam2W_T)
            pcl_list.append(o3d_cam_pose)
        return pcl_list

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_root", default="data/replica/room_0")
    argparser.add_argument("--img_wh", nargs="+", type=int, default=[1280, 720])
    argparser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')
    argparser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    args = argparser.parse_args()
    kwargs = {'root_dir': args.data_root,
                'img_wh': tuple(args.img_wh)}
    kwargs['spheric_poses'] = args.spheric_poses
    kwargs['val_num'] = args.num_gpus
    train_dataset = CustomLLFFDataset(split='train', **kwargs)

    # train_dataloader = DataLoader(train_dataset, shuffle=False,
    #                         num_workers=1, batch_size=1,pin_memory=True)

    pcl_list = train_dataset.get_pcls_from_rgbds()
    show_pcl_list(pcl_list)

if __name__=="__main__":
    main()
