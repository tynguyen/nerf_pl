ROOT_DIR="data/replica/room_0"
# directory containing the data
IMG_W=1280  # image width (do not set too large)
IMG_H=720  # image height (do not set too large)
NUM_EPOCHS=30  # number of epochs to train (depending on how many images there are,
# 20~30 might be enough)
EXP="exp_replica_room_0"  # name of the experience (arbitrary)

python3 train.py \
   --dataset_name llff \
   --root_dir "$ROOT_DIR" \
   --N_importance 64 --img_wh $IMG_W $IMG_H \
   --num_epochs $NUM_EPOCHS --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler cosine \
   --exp_name $EXP
