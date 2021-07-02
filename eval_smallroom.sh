python eval.py \
   --root_dir data/july_05th/smallroom_cam \
   --dataset_name llff --scene_name lego \
   --spheric_poses --use_disp \
   --img_wh 504 378 --N_importance 64 --ckpt_path 'ckpts/exp_small_room/epoch=09.ckpt' \