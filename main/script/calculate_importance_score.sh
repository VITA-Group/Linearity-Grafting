python -u calculate_importance_score.py \
    --dataset cifar10 \
    --arch cifar_cnn_b_hook_pre \
    --weight_dir Train_FAT_cifar_cnn_b_seed1_2_255/model_RA_best.pth.tar \
    --seed 1 \
    --save_dir Train_FAT_cifar_cnn_b_seed1_2_255 \
    --mode magnitude


