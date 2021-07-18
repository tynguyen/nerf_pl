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
    o3d.visualization.draw_geometries(pcl_list)

def get_pcl_from_rgbd_in_opengl_frame(rgb, depth, K):
    """Get a point cloud in the OpenGL camera frame from RGBD

    Args:
        rgb ([HxW x 3]): rgb image
        depth ([HxW]): depth image
        K ([3x3]): intrinsics parameters
    """
    # Get 3D points in the OpenCV camera's coord
    depth = depth.squeeze()[None]  # 1 x H x W
    img_h, img_w = rgb.shape[:2]
    # Note that in opengl convention, y is up, x is to the right. Therefore,
    # y start from img_h-1 to 0 from the upper left corner.
    xs, ys = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(img_h - 1, 0, img_h),
    )
    xs = xs.reshape(1, img_h, img_w)
    ys = ys.reshape(1, img_h, img_w)

    # Unproject (OpenGL camera's coordinate). Negate depth value because Z of the camera is backward
    xys = np.vstack((-xs * depth, -ys * depth, -depth))#, np.ones(depth.shape)))
    xys = xys.reshape(3, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    # Visualize the points
    pcl_points = xy_c0[:3, :].T
    pcl_cam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_points))
    return pcl_cam

class CustomLLFFDataset(LLFFDataset):
    def read_meta(self):
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                        # load full resolution image then resize
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)

        self.image_names = [name.split('/')[-1] for name in self.image_paths]

        image_names_file = os.path.join(self.root_dir,
                    'image_names_corresponding_to_poses_bounds.txt')
        if os.path.exists(image_names_file):
            with open(image_names_file, 'r') as fin:
                image_names = fin.readlines() # (N_images)
                image_names = [name.strip() for name in image_names]
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

        # Camera intrinsics (OpenGL frame)
        # In OpenGL frame, Z is flipped so we need to negate focal value
        self.gl_intrinsics = np.array([
            [-self.focal,    0,       self.img_wh[0]/2],
            [0         , -self.focal, self.img_wh[1]/2],
            [0         ,    0,            1]])
        print(f"[Info] Intrinsics: \n {self.gl_intrinsics}")
        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        self.poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal

    def get_pcls_from_rgbds(self, max_no_pcls=1e10):
        """Get pointclouds from RGB + Depth images
        """
        pcl_list = []
        rgbd_images = read_rgbd_images(self.root_dir, self.image_names, rgb_ext='.jpg', depth_ext='.png')

        for i, image_name in enumerate(self.image_names):
            if i >= max_no_pcls:
                break
            # if image_name != "nodeID_8_angleID_0.jpg":
            #     continue
            print(f"[Info]--> Frame {image_name}")
            rgb, depth = rgbd_images[image_name]
            # plt.subplot(121)
            # plt.imshow(depth, cmap="jet")
            depth = depth / 1000.0
            depth = depth.astype(np.float32)
            # plt.subplot(122)
            # plt.imshow(depth, cmap="jet")
            # plt.show()
            glCam2W_T = self.poses[i] # 3x4
            glCam2W_T = np.vstack([glCam2W_T, np.array([0,0,0,1])])

            # Point cloud. Note that we cannot use Open3D pcl from RGBD anymore because our camera now is
            # OpenGL model.
            # o3d_rgbd_pcl = getPCLfromRGB_D(rgb, depth, self.cv_intrinsics) # Wrong!
            o3d_rgbd_pcl = get_pcl_from_rgbd_in_opengl_frame(rgb, depth, self.gl_intrinsics)

            # Camera pose
            o3d_cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

            # Transform to the world
            print(f"[Info] Transform from OpenGL cam to world:\n {glCam2W_T}")
            o3d_rgbd_pcl.transform(glCam2W_T)
            o3d_cam_pose.transform(glCam2W_T)
            pcl_list.append(o3d_rgbd_pcl)
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
