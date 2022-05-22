CUDA_VISIBLE_DEVICES=0 python -u train_grafting.py \
    --dataset cifar10 \
    --arch cifar_cnn_b_graft \
    --seed 1 \
    --weight_dir Train_FAT_cifar_cnn_b_seed1_2_255/model_RA_best.pth.tar \
    --mask_dir mask/FAT_CNNB_eps2_255_graft_50_cifar.pt \
    --save_dir GRAFT_FAT_cifar_cnn_b_graft_seed1_2_255_graft_50 