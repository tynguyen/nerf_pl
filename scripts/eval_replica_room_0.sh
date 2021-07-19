# directory containing the data
ROOT_DIR="data/replica/room_0"
EXP_DIR="ckpts"
IMG_W=640  # image width (do not set too large)
IMG_H=360  # image height (do not set too large)

EXP="exp_replica_room_0_spheric"  # name of the experience (arbitrary)
EPOCH="3"  # name of the experience (arbitrary)


python eval.py \
   --root_dir "$ROOT_DIR" \
   --dataset_name llff --scene_name room_0 \
   --spheric_poses --use_disp \
   --img_wh ${IMG_W} ${IMG_H} --N_importance 256 --ckpt_path ${EXP_DIR}/${EXP}/epoch=${EPOCH}.ckpt \
