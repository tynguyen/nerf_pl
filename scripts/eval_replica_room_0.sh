# directory containing the data
ROOT_DIR="data/replica/room_0_array"
EXP_DIR="ckpts"
IMG_W=640  # image width (do not set too large)
IMG_H=360  # image height (do not set too large)

EXP="replica_room_0_array"  # name of the experience (arbitrary)
EPOCH="29"  # name of the experience (arbitrary)


python eval.py \
   --root_dir "$ROOT_DIR" \
   --dataset_name llff --scene_name room_0 \
   --img_wh ${IMG_W} ${IMG_H} --N_importance 64 --ckpt_path ${EXP_DIR}/${EXP}/epoch=${EPOCH}.ckpt \
