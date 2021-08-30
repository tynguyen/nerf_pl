ROOT_DIR="data/real_world_data/small_room_4_mics"
# directory containing the data
IMG_W=640  # image width (do not set too large)
IMG_H=360  # image height (do not set too large)
NUM_EPOCHS=30  # number of epochs to train (depending on how many images there are,
# 20~30 might be enough)
EXP="real_world_data_small_room_4_mics"  # name of the experience (arbitrary)

# Normalize the sampled 3D points to the unit sphere without using NDC
NORMALIZE_SAMPLED_POINTS="True"
RAYS_SCALE_FACTOR=0.065
RAY_TO_NDC="False"
python3 train.py \
   --dataset_name llff \
   --root_dir "$ROOT_DIR" \
   --N_importance 64 --img_wh $IMG_W $IMG_H \
   --num_epochs $NUM_EPOCHS --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --normalize_sampled_points $NORMALIZE_SAMPLED_POINTS\
   --rays_scale_factor $RAYS_SCALE_FACTOR\
   --center_3dpoints -0.0028 0.0005 -0.3630\
   --ray_to_NDC $RAY_TO_NDC\
   --lr_scheduler cosine \
   --exp_name $EXP
