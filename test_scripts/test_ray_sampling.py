import os, sys
from typing import NamedTuple
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from nerf_pl.opt import get_opts
from nerf_pl.datasets import dataset_dict
from nerf_pl.utils import *

# optimizer, scheduler, visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def stratified_uniform_sampling(
    rays: torch.Tensor, N_samples: int = 64
) -> torch.Tensor:
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + far * z_steps

    z_vals = z_vals.expand(N_rays, N_samples)

    z_vals_mid = 0.5 * (
        z_vals[:, :-1] + z_vals[:, 1:]
    )  # (N_rays, N_samples-1) interval mid points
    # get intervals between samples
    upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
    lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)
    perturb_rand = torch.rand(z_vals.shape, device=rays.device)
    z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
        2
    )  # (N_rays, N_samples, 3)
    return xyz_coarse_sampled.view(-1, 3)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def prepare_data(hparams):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {
        "root_dir": hparams.root_dir,
        "img_wh": tuple(hparams.img_wh),
    }
    if hparams.dataset_name == "llff":
        kwargs["spheric_poses"] = hparams.spheric_poses
        kwargs["val_num"] = hparams.num_gpus
        kwargs["use_NDC"] = hparams.ray_to_NDC
        kwargs["normalize_sampled_points"] = hparams.normalize_sampled_points
    train_dataset = dataset(split="train", **kwargs)
    return train_dataset


def visual_points(ax: mplot3d.Axes3D, points: torch.Tensor, colors: np.ndarray) -> None:
    """ Visualize points
    @param ax: matplotlib axes
    @param points: (N, 3)
    @param colors: (N, 3)
    """
    points = points.numpy()
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], alpha=0.3, marker=".", s=20, c=colors
    )


def visual_rays(ax: plt.axes, rays: torch.Tensor, colors: np.ndarray = None) -> None:
    """
    Visualize rays
    @param ax: matplotlib axes
    @param rays: (N, 8): rays in N x [ox, oy, oz, dx, dy, dz, near, far] format
    @param colors: (N, 3): RGB colors for each ray
    """
    # Get the rays
    ox, oy, oz, dx, dy, dz, near, far = torch.split(rays, 1, dim=1)  # (N, 1) each

    # Get the colors
    if colors is None:
        colors = torch.cat((ox, oy, oz), dim=1)
        colors = torch.abs(colors) / torch.norm(colors, dim=1, keepdim=True)
        colors = colors.numpy()

    # Get the points
    points = torch.cat((ox, oy, oz), dim=1)
    points = points.numpy()

    # Get the directions
    directions = torch.cat((dx, dy, dz), dim=1)
    directions = directions.numpy()

    # Get the near and far
    near = near.numpy()
    far = far.numpy()
    # Add
    directions = directions * far  # (far - near)

    # Plot the rays
    # Plot origins
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], alpha=0.5, marker="s", s=45, c=colors
    )
    # Plot ending points
    ax.scatter(
        points[:, 0] + directions[:, 0],
        points[:, 1] + directions[:, 1],
        points[:, 2] + directions[:, 2],
        marker=".",
        s=1,
        alpha=0.5,
        color=colors,
    )
    for i in range(len(points)):
        a = Arrow3D(
            [points[i, 0], points[i, 0] + directions[i, 0]],
            [points[i, 1], points[i, 1] + directions[i, 1]],
            [points[i, 2], points[i, 2] + directions[i, 2]],
            mutation_scale=10,
            lw=1,
            arrowstyle="->",
            color=colors[i],
        )
        ax.add_artist(a)


def test_ray_sampling(hparams: NamedTuple):
    if hparams is None:
        return
    dataset = prepare_data(hparams)
    print(f"[Info] Get dataset prepared!")

    # Image size
    img_w, img_h = hparams.img_wh[0], hparams.img_wh[1]

    # Rays and RGB colors
    # Each ray is a 8-dim vector: [ox, oy, oz, dx, dy, dz, near, far]
    all_rays = dataset.all_rays  # ((N_images-1)*h*w, 8)
    all_rgbs = dataset.all_rgbs  # ((N_images-1)*h*w, 3)

    # Number of images
    num_images = all_rays.shape[0] // (img_w * img_h) + 1
    print(f"[Info] Number of images: {num_images}")

    # Create a figure
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Plot the rays
    for i in range(num_images):
        color = np.random.rand(1, 3)
        rays = all_rays[
            i * img_w * img_h : (i + 1) * img_w * img_h
        ]  # Limit to 100 lines
        print(f"[Info] cam {i+1}th| totally {len(rays)} rays")
        perm = torch.randperm(rays.size(0))
        idx = perm[:5]
        rays = rays[idx]
        colors = np.tile(color, (rays.size(0), 1))
        visual_rays(ax, rays, colors)
        print(f"[Info] cam {i+1}th| Display {len(rays)} rays")

        # Sample N_sample points per ray
        xyz_sampled = stratified_uniform_sampling(rays, hparams.N_samples)
        print(f"[Info] cam {i+1}th| sampled {hparams.N_samples} points")
        colors = np.tile(color, (xyz_sampled.size(0), 1))
        visual_points(ax, xyz_sampled, colors)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.autoscale(enable=False, axis="both", tight=True)
    plt.title(
        f"Rays with NDC = {hparams.ray_to_NDC} | normalize sampled points = {hparams.normalize_sampled_points}"
    )
    plt.savefig(
        f"assets/rays_with_NDC={hparams.ray_to_NDC}_normalize_sampled_points={hparams.normalize_sampled_points}.png"
    )
    plt.show()

    if hparams.normalize_sampled_points:
        print(f"[Info] Ray scale factor: {dataset.rays_scale_factor}")
        print(
            f"[Info] Center point of all 3D sampled points: {dataset.center_3dpoints}"
        )


if __name__ == "__main__":
    hparams = get_opts()
    ROOT_DIR = "data/replica/room_0_array"
    # directory containing the data
    IMG_W = 640  # image width (do not set too large)
    IMG_H = 360  # image height (do not set too large)
    hparams.dataset_name = "llff"
    hparams.root_dir = ROOT_DIR
    hparams.img_wh = (IMG_W, IMG_H)
    hparams.N_samples = 10
    hparams.split = "val"

    # # Test ray sampling with NDC
    hparams.ray_to_NDC = True  # Use NDC
    test_ray_sampling(hparams)

    # Test ray sampling without NDC
    hparams.ray_to_NDC = False  # NOT Use NDC
    test_ray_sampling(hparams)

    # Test ray sampling without NDC but normalize_sampled_points
    hparams.ray_to_NDC = False  # NOT Use NDC
    hparams.normalize_sampled_points = True
    test_ray_sampling(hparams)
