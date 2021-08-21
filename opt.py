import argparse


def str2bool(s):
    return False if s.lower() == "false" else True


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
        help="root directory of dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=["blender", "llff"],
        help="which dataset to train/val",
    )
    parser.add_argument(
        "--img_wh",
        nargs="+",
        type=int,
        default=[800, 800],
        help="resolution (img_w, img_h) of the image",
    )
    parser.add_argument(
        "--spheric_poses",
        default=False,
        action="store_true",
        help="whether images are taken in spheric poses (for llff)",
    )

    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=128,
        help="number of additional fine samples",
    )
    parser.add_argument(
        "--use_disp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--ray_to_NDC",
        default=True,
        type=str2bool,
        help="Convert the camera poses to the NDC? By default, False. This only affects when --spheric_poses is False",
    )
    parser.add_argument(
        "--normalize_sampled_points",
        default=False,
        type=str2bool,
        help="Convert the camera poses to [-1, 1] range? By default, True. This only affects when --spheric_poses is False",
    )
    parser.add_argument(
        "--center_3dpoints",
        default=[-0.0028, 0.0005, -0.3630],
        type=float,
        nargs="+",
        help="Center of sampled 3D points. This is used with --normalize_sampled_points set True",
    )
    parser.add_argument(
        "--rays_scale_factor",
        default=0.065,
        type=float,
        help="Multiplication factor to 3D points before centering to make their coordinates to be within [-1, 1]. This is used with --normalize_sampled_points set True",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="factor to perturb depth sampling points",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="std dev of noise added to regularize sigma",
    )

    parser.add_argument(
        "--loss_type", type=str, default="mse", choices=["mse"], help="loss to use"
    )

    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument(
        "--chunk",
        type=int,
        default=32 * 1024,
        help="chunk size to split the input to avoid OOM",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=16, help="number of training epochs"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")

    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="pretrained checkpoint path to load"
    )
    parser.add_argument(
        "--prefixes_to_ignore",
        nargs="+",
        type=str,
        default=["loss"],
        help="the prefixes to ignore in the checkpoint state dict",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer type",
        choices=["sgd", "adam", "radam", "ranger"],
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="learning rate momentum"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="steplr",
        help="scheduler type",
        choices=["steplr", "cosine", "poly"],
    )
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument(
        "--warmup_multiplier",
        type=float,
        default=1.0,
        help="lr is multiplied by this factor after --warmup_epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Gradually warm-up(increasing) learning rate in optimizer",
    )
    ###########################
    #### params for steplr ####
    parser.add_argument(
        "--decay_step", nargs="+", type=int, default=[20], help="scheduler decay step"
    )
    parser.add_argument(
        "--decay_gamma", type=float, default=0.1, help="learning rate decay amount"
    )
    ###########################
    #### params for poly ####
    parser.add_argument(
        "--poly_exp",
        type=float,
        default=0.9,
        help="exponent for polynomial learning rate decay",
    )
    ###########################

    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")

    return parser.parse_args()
